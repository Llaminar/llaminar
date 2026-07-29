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
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

extern "C" bool cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
    uint8_t codebook_id,
    int n,
    int k,
    int sm_count,
    int *uses_ordered_reducer,
    int *k_partitions);

// =========================================================================
// Per-device prefill context — replaces process-global statics for SM count
// cache and stream-K fixup buffer.  Owned by KernelFactory, one per device.
// =========================================================================
struct CUDAPrefillContext_
{
    int sm_count = 0;
    int device_id = -1;
    float *workspace_splitk_partials = nullptr;
    size_t workspace_splitk_partials_size = 0;
    float *workspace_fixup_buf = nullptr;
    size_t workspace_fixup_buf_size = 0;
};

struct LastLaunchSelection_
{
    int tile_id = -1;
    int split_k = 1;
    int used_bk256 = 0;
    int used_streamk = 0;
};

// Thread-local because tile sweep benchmarks can launch from multiple worker
// threads. Diagnostics read back the selection on the same thread.
static thread_local LastLaunchSelection_ g_last_launch_selection;

static inline void recordLastLaunchSelection(
    int tile_id,
    int split_k,
    bool used_bk256,
    int used_streamk)
{
    g_last_launch_selection.tile_id = tile_id;
    g_last_launch_selection.split_k = split_k;
    g_last_launch_selection.used_bk256 = used_bk256 ? 1 : 0;
    g_last_launch_selection.used_streamk = used_streamk;
}

static int querySmCount(CUDAPrefillContext_ *ctx)
{
    if (ctx->sm_count > 0)
        return ctx->sm_count;
    cudaDeviceGetAttribute(&ctx->sm_count,
                           cudaDevAttrMultiProcessorCount,
                           ctx->device_id);
    if (ctx->sm_count <= 0)
        ctx->sm_count = 82;
    return ctx->sm_count;
}

static float *getOrAllocFixupBuffer(CUDAPrefillContext_ *ctx, size_t required_bytes, cudaStream_t stream)
{
    if (ctx->workspace_fixup_buf && ctx->workspace_fixup_buf_size >= required_bytes)
    {
        cudaMemsetAsync(ctx->workspace_fixup_buf, 0, required_bytes, stream);
        return ctx->workspace_fixup_buf;
    }
    return nullptr;
}

static float *getOrAllocSplitkPartials(CUDAPrefillContext_ *ctx, size_t required_bytes, cudaStream_t stream)
{
    // Split-K reducers consume every partition slot. Some legal dispatches have
    // trailing empty K partitions, so stale workspace contents must not survive
    // from an earlier request or projection.
    if (ctx->workspace_splitk_partials && ctx->workspace_splitk_partials_size >= required_bytes)
    {
        cudaMemsetAsync(ctx->workspace_splitk_partials, 0, required_bytes, stream);
        return ctx->workspace_splitk_partials;
    }
    return nullptr;
}

namespace
{
    using llaminar2::cuda_native_vnni::fp16_bits_to_float;

    // BK=64 constants (CUTLASS-standard K-tile for INT8 on SM80+)
    constexpr int BK64 = 64;
    constexpr int SMEM_PAD_64 = 16;
    [[maybe_unused]] constexpr int SMEM_STRIDE_64 = BK64 + SMEM_PAD_64; // 80, 16-byte aligned for ldmatrix

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

    // ─── Split-K two-phase reduce kernel ───────────────────────────────
    // Sums SPLIT_K partial results into the final output buffer C.
    // Each z-slice of the GEMM wrote its partial to partials[z * M * N].
    // This kernel sums across z-slices for each (m, n) element.
    // Also applies beta * C_existing + bias if present.
    // Grid: ceil(M*N / 256), Block: 256
    __global__ void splitk_reduce(
        const float *__restrict__ partials,
        float *__restrict__ C,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int M, int N, int split_k,
        float beta)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = M * N;
        if (idx >= total)
            return;

        float sum = 0.0f;
        for (int z = 0; z < split_k; ++z)
            sum += partials[z * total + idx];

        if (beta != 0.0f && C_existing)
            sum += beta * C_existing[idx];
        if (bias)
            sum += bias[idx % N];

        C[idx] = sum;
    }

    [[maybe_unused]] __device__ __forceinline__ int frag_row(int lane_id, int elem)
    {
        return (elem >> 1) * 8 + (lane_id >> 2);
    }

    [[maybe_unused]] __device__ __forceinline__ int frag_col(int lane_id, int elem)
    {
        return (lane_id & 3) * 2 + (elem & 1);
    }

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

    // Named barrier: sync a subset of threads within a CTA.
    // barrier_id: 0-15 (hardware barrier slot), thread_count: participating threads.
    // Has acquire/release memory ordering (like __syncthreads but scoped to participants).
    [[maybe_unused]] __device__ __forceinline__ void named_bar_sync(int barrier_id, int thread_count)
    {
        asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(thread_count) : "memory");
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

    [[maybe_unused]] __device__ __forceinline__ void mma_m16n8k32_s8(
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
              bool FIXUP_TWO_PASS = false,
              int BLOCK_SIZE_ = WARPS_M * WARPS_N * 32,
              // Stream-K tile loop: the outer while-loop keeps all global-memory
              // pointer registers (~16 GP regs) alive across iterations, requiring
              // ~124 regs total. With MIN_BLOCKS=2 (64-reg cap), this causes
              // catastrophic spilling. MIN_BLOCKS=1 (128-reg cap) avoids spills.
              //
              // COMPILER INSIGHT: A bounded for-loop (e.g. `for(i=0;i<4;i++)`) with
              // compile-time-constant upper bound achieves REG:64 STACK:0 — the
              // compiler can drop/reload pointers from ld.param between iterations.
              // However, the k-loop compute pipeline itself degrades at 64 regs:
              // 32 accumulator regs (WM=2×WN=4×4) leave only 32 for pipeline state,
              // shared memory pointers, and ILP — resulting in 2× slower per-tile
              // throughput that 2× occupancy cannot overcome.
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
        const int32_t *__restrict__ sums_A,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        int serial_m1_k_partitions,
        int serial_m1_uses_ordered_reducer,
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
        [[maybe_unused]] constexpr bool USE_NAMED_BARRIERS = false; // (A_VEC_LOADS == BLOCK_SIZE) && (BLOCK_SIZE >= 2 * BN);

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0);
        static_assert(WARP_N % 8 == 0);
        static_assert(!STREAM_K || SPLIT_K == 1, "Stream-K and Split-K are mutually exclusive");
        static_assert(!FIXUP_TWO_PASS || STREAM_K, "FIXUP_TWO_PASS requires STREAM_K");

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
        float serial_acc[WM][WN][4];

        /**
         * Fold a completed K partition with the same ascending FP32 reducer
         * used by public M=1 KPAR decode.
         *
         * The tensor-core CTA still owns all K work for several rows, so this
         * adds no global partial buffer or extra launch. It merely preserves
         * the serial policy's parenthesization in private registers.
         */
        auto commit_serial_m1_partition =
            [&](int completed_block) __attribute__((always_inline))
        {
            if constexpr (STREAM_K || SPLIT_K != 1)
                return;
            if (!serial_m1_uses_ordered_reducer)
                return;

            const int blocks_per_partition =
                (num_q40_blocks + serial_m1_k_partitions - 1) /
                serial_m1_k_partitions;
            const bool partition_complete =
                completed_block == num_q40_blocks ||
                (completed_block % blocks_per_partition) == 0;
            if (!partition_complete)
                return;

#pragma unroll
            for (int i = 0; i < WM; ++i)
#pragma unroll
                for (int j = 0; j < WN; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        /*
                         * Serial KPAR scales every partition before reducing
                         * it. Applying alpha after the fold is observably
                         * different for non-unit alpha.
                         */
                        const float partial =
                            __fmul_rn(alpha, acc[i][j][e]);
                        serial_acc[i][j][e] = __fadd_rn(
                            serial_acc[i][j][e],
                            partial);
                        acc[i][j][e] = 0.0f;
                    }
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
                else
                {
                    /*
                     * `ldmatrix` fetches a complete physical fragment even when
                     * only one semantic output column remains. Every invalid
                     * column must therefore be initialized. Leaving gcol>=N
                     * untouched let stale shared-memory bytes participate in
                     * the valid border-column MMA result.
                     */
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

                /*
                 * Preserve the exact FP16 metadata bits until the shared
                 * decode-equivalent contribution helper consumes them. Earlier
                 * versions expanded these values and then re-expressed every
                 * format's FP32 formula locally, allowing the tensor-core path
                 * to acquire a different contraction tree from serial DP4A.
                 */
                uint16_t scale_bits_pre[WN][2];
                [[maybe_unused]] uint16_t secondary_bits_pre[WN][2];
                [[maybe_unused]] uint32_t emin_bits_pre[WN][2];
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
                {
                    const int b_col_base = wc * WARP_N + wj * 8;
                    scale_bits_pre[wj][0] =
                        smem_scales_B[stage][2 * (b_col_base + frag_col(lane_id, 0)) + scale_slot];
                    scale_bits_pre[wj][1] =
                        smem_scales_B[stage][2 * (b_col_base + frag_col(lane_id, 1)) + scale_slot];
                    if constexpr (NEEDS_MINS)
                    {
                        secondary_bits_pre[wj][0] =
                            smem_mins_B[stage][2 * (b_col_base + frag_col(lane_id, 0)) + scale_slot];
                        secondary_bits_pre[wj][1] =
                            smem_mins_B[stage][2 * (b_col_base + frag_col(lane_id, 1)) + scale_slot];
                    }
                    if constexpr (IS_DUAL_SCALE_ASYM)
                    {
                        emin_bits_pre[wj][0] =
                            smem_emins_B[stage][2 * (b_col_base + frag_col(lane_id, 0)) + scale_slot];
                        emin_bits_pre[wj][1] =
                            smem_emins_B[stage][2 * (b_col_base + frag_col(lane_id, 1)) + scale_slot];
                    }
                }

                // ── Pre-load ALL A fragments for this half ────────────
                // Decouples A ldmatrix from MMA, enabling the restructured
                // wj→wi loop that loads each B fragment ONCE instead of WM times.
                uint32_t A_frag_all[WM][4];
                [[maybe_unused]] int sum_A_row0_all[WM], sum_A_row1_all[WM];
                [[maybe_unused]] int sum_A_lo_row0_all[WM], sum_A_lo_row1_all[WM];
                [[maybe_unused]] int sum_A_hi_row0_all[WM], sum_A_hi_row1_all[WM];
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
                        const int grow0 = block_m + a_row_base + gid;
                        const int grow1 = grow0 + 8;
                        if (sums_A)
                        {
                            sum_A_row0_all[wi] =
                                sums_A[static_cast<size_t>(grow0) * num_q40_blocks + kb];
                            sum_A_row1_all[wi] =
                                sums_A[static_cast<size_t>(grow1) * num_q40_blocks + kb];
                        }
                        else
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
                            sum_A_row0_all[wi] = s0;
                            sum_A_row1_all[wi] = s1;
                        }
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
                        sum_A_lo_row0_all[wi] = slo0;
                        sum_A_lo_row1_all[wi] = slo1;
                        sum_A_hi_row0_all[wi] = shi0;
                        sum_A_hi_row1_all[wi] = shi1;
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

                        int32_t dot_lo[4] = {0, 0, 0, 0};
                        int32_t dot_hi[4] = {0, 0, 0, 0};
                        if constexpr (IS_DUAL_SCALE)
                        {
                            const uint32_t B_lo[2] = {B_frag[0], 0u};
                            const uint32_t B_hi[2] = {0u, B_frag[1]};
                            mma_m16n8k32_s8(
                                dot_lo,
                                A_frag_all[wi],
                                B_lo);
                            mma_m16n8k32_s8(
                                dot_hi,
                                A_frag_all[wi],
                                B_hi);
                        }
                        else
                        {
                            mma_m16n8k32_s8(
                                dot_lo,
                                A_frag_all[wi],
                                B_frag);
                        }

                        const int b_col_base =
                            wc * WARP_N + wj * 8;
                        const uint16_t qh_c0 = IS_IQ1_M
                            ? smem_iq1m_qh[stage][
                                  2 * (b_col_base +
                                       frag_col(lane_id, 0)) +
                                  scale_slot]
                            : uint16_t{0};
                        const uint16_t qh_c1 = IS_IQ1_M
                            ? smem_iq1m_qh[stage][
                                  2 * (b_col_base +
                                       frag_col(lane_id, 1)) +
                                  scale_slot]
                            : uint16_t{0};

#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            const int column = e & 1;
                            const bool first_row = e < 2;
                            const float activation_scale =
                                first_row ? sa0 : sa1;
                            const int activation_sum =
                                IS_ASYMMETRIC
                                    ? (first_row
                                           ? sum_A_row0_all[wi]
                                           : sum_A_row1_all[wi])
                                    : 0;
                            const int activation_sum_lo =
                                (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                    ? (first_row
                                           ? sum_A_lo_row0_all[wi]
                                           : sum_A_lo_row1_all[wi])
                                    : 0;
                            const int activation_sum_hi =
                                (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                    ? (first_row
                                           ? sum_A_hi_row0_all[wi]
                                           : sum_A_hi_row1_all[wi])
                                    : 0;
                            const uint32_t emin_bits =
                                IS_DUAL_SCALE_ASYM
                                    ? emin_bits_pre[wj][column]
                                    : uint32_t{0};
                            const uint16_t secondary_bits =
                                NEEDS_MINS
                                    ? secondary_bits_pre[wj][column]
                                    : uint16_t{0};
                            const uint16_t iq1m_qh =
                                column ? qh_c1 : qh_c0;
                            const int subgroup_sum0 = IS_IQ1_M
                                ? (first_row
                                       ? sg0_r0_all[wi]
                                       : sg0_r1_all[wi])
                                : 0;
                            const int subgroup_sum1 = IS_IQ1_M
                                ? (first_row
                                       ? sg1_r0_all[wi]
                                       : sg1_r1_all[wi])
                                : 0;
                            const int subgroup_sum2 = IS_IQ1_M
                                ? (first_row
                                       ? sg2_r0_all[wi]
                                       : sg2_r1_all[wi])
                                : 0;
                            const int subgroup_sum3 = IS_IQ1_M
                                ? (first_row
                                       ? sg3_r0_all[wi]
                                       : sg3_r1_all[wi])
                                : 0;
                            const float contribution =
                                llaminar2::cuda_native_vnni::
                                    native_vnni_block_contribution_from_reduced_terms_rn<
                                        CODEBOOK_ID>(
                                        dot_lo[e],
                                        dot_hi[e],
                                        activation_scale,
                                        scale_bits_pre[wj][column],
                                        secondary_bits,
                                        emin_bits,
                                        activation_sum,
                                        activation_sum_lo,
                                        activation_sum_hi,
                                        iq1m_qh,
                                        subgroup_sum0,
                                        subgroup_sum1,
                                        subgroup_sum2,
                                        subgroup_sum3);
                            acc[wi][wj][e] = __fadd_rn(
                                acc[wi][wj][e],
                                contribution);
                        }
                    }
                }
                commit_serial_m1_partition(kb + 1);
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
                    [[maybe_unused]] int sum_A_row0 = 0, sum_A_row1 = 0;
                    if constexpr (IS_ASYMMETRIC)
                    {
                        if (sums_A)
                        {
                            if (grow0 < M)
                                sum_A_row0 =
                                    sums_A[static_cast<size_t>(grow0) * num_q40_blocks + kb];
                            if (grow1 < M)
                                sum_A_row1 =
                                    sums_A[static_cast<size_t>(grow1) * num_q40_blocks + kb];
                        }
                        else if (grow0 < M)
                        {
                            const int8_t *row0_ptr = &smem_A[stage][(a_row_base + gid) * SMEM_STRIDE_64 + k_offset];
                            int32_t s0 = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w)
                                s0 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row0_ptr)[w], s0);
                            sum_A_row0 = s0;
                            if (grow1 < M)
                            {
                                const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                                int32_t s1 = 0;
#pragma unroll
                                for (int w = 0; w < 8; ++w)
                                    s1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], s1);
                                sum_A_row1 = s1;
                            }
                        }
                        else if (grow1 < M)
                        {
                            const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                            int32_t s1 = 0;
#pragma unroll
                            for (int w = 0; w < 8; ++w)
                                s1 = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], s1);
                            sum_A_row1 = s1;
                        }
                    }

                    // Split sums for dual_scale_asym (Q2_K) and IQ1_M
                    [[maybe_unused]] int sum_A_lo_row0 = 0, sum_A_lo_row1 = 0;
                    [[maybe_unused]] int sum_A_hi_row0 = 0, sum_A_hi_row1 = 0;
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
                            sum_A_lo_row0 = slo;
                            sum_A_hi_row0 = shi;
                        }
                        if (grow1 < M)
                        {
                            const int8_t *row1_ptr = &smem_A[stage][(a_row_base + gid + 8) * SMEM_STRIDE_64 + k_offset];
                            int32_t slo = 0, shi = 0;
                            for (int w = 0; w < 4; ++w)
                                slo = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], slo);
                            for (int w = 4; w < 8; ++w)
                                shi = __dp4a(0x01010101, reinterpret_cast<const int32_t *>(row1_ptr)[w], shi);
                            sum_A_lo_row1 = slo;
                            sum_A_hi_row1 = shi;
                        }
                    }

                    [[maybe_unused]] int subgroup_r0[4] = {0, 0, 0, 0};
                    [[maybe_unused]] int subgroup_r1[4] = {0, 0, 0, 0};
                    if constexpr (IS_IQ1_M)
                    {
                        if (grow0 < M)
                        {
                            const int32_t *row0_words =
                                reinterpret_cast<const int32_t *>(
                                    &smem_A[stage][
                                        (a_row_base + gid) *
                                            SMEM_STRIDE_64 +
                                        k_offset]);
#pragma unroll
                            for (int word = 0; word < 8; ++word)
                            {
                                subgroup_r0[word / 2] +=
                                    llaminar2::cuda_native_vnni::
                                        sum_packed_i8(row0_words[word]);
                            }
                        }
                        if (grow1 < M)
                        {
                            const int32_t *row1_words =
                                reinterpret_cast<const int32_t *>(
                                    &smem_A[stage][
                                        (a_row_base + gid + 8) *
                                            SMEM_STRIDE_64 +
                                        k_offset]);
#pragma unroll
                            for (int word = 0; word < 8; ++word)
                            {
                                subgroup_r1[word / 2] +=
                                    llaminar2::cuda_native_vnni::
                                        sum_packed_i8(row1_words[word]);
                            }
                        }
                    }

#pragma unroll
                    for (int wj = 0; wj < WN; ++wj)
                    {
                        const int b_col_base = wc * WARP_N + wj * 8;
                        const int lc0 = b_col_base + frag_col(lane_id, 0);
                        const int lc1 = b_col_base + frag_col(lane_id, 1);
                        const bool column0_valid = block_n + lc0 < N;
                        const bool column1_valid = block_n + lc1 < N;
                        const uint16_t scale_bits[2] = {
                            column0_valid
                                ? smem_scales_B[stage][2 * lc0 + scale_slot]
                                : uint16_t{0},
                            column1_valid
                                ? smem_scales_B[stage][2 * lc1 + scale_slot]
                                : uint16_t{0}};
                        const uint16_t secondary_bits[2] = {
                            NEEDS_MINS && column0_valid
                                ? smem_mins_B[stage][2 * lc0 + scale_slot]
                                : uint16_t{0},
                            NEEDS_MINS && column1_valid
                                ? smem_mins_B[stage][2 * lc1 + scale_slot]
                                : uint16_t{0}};
                        const uint32_t emin_bits[2] = {
                            IS_DUAL_SCALE_ASYM && column0_valid
                                ? smem_emins_B[stage][2 * lc0 + scale_slot]
                                : uint32_t{0},
                            IS_DUAL_SCALE_ASYM && column1_valid
                                ? smem_emins_B[stage][2 * lc1 + scale_slot]
                                : uint32_t{0}};
                        const uint16_t iq1m_qh[2] = {
                            IS_IQ1_M && column0_valid
                                ? smem_iq1m_qh[stage][2 * lc0 + scale_slot]
                                : uint16_t{0},
                            IS_IQ1_M && column1_valid
                                ? smem_iq1m_qh[stage][2 * lc1 + scale_slot]
                                : uint16_t{0}};

                        uint32_t B_frag[2];
                        load_ldmatrix_b_m16n8k32(
                            B_frag,
                            reinterpret_cast<const int *>(&smem_B[stage][b_col_base * SMEM_STRIDE_64 + k_offset]),
                            SMEM_STRIDE_64 / 4, lane_id);

                        int32_t dot_lo[4] = {0, 0, 0, 0};
                        int32_t dot_hi[4] = {0, 0, 0, 0};
                        if constexpr (IS_DUAL_SCALE)
                        {
                            const uint32_t B_lo[2] = {B_frag[0], 0u};
                            const uint32_t B_hi[2] = {0u, B_frag[1]};
                            mma_m16n8k32_s8(dot_lo, A_frag, B_lo);
                            mma_m16n8k32_s8(dot_hi, A_frag, B_hi);
                        }
                        else
                        {
                            mma_m16n8k32_s8(dot_lo, A_frag, B_frag);
                        }

#pragma unroll
                        for (int e = 0; e < 4; ++e)
                        {
                            const int column = e & 1;
                            const bool first_row = e < 2;
                            const float activation_scale =
                                first_row ? sa0 : sa1;
                            const int *subgroup_sums =
                                first_row ? subgroup_r0 : subgroup_r1;
                            const float contribution =
                                llaminar2::cuda_native_vnni::
                                    native_vnni_block_contribution_from_reduced_terms_rn<
                                        CODEBOOK_ID>(
                                        dot_lo[e],
                                        dot_hi[e],
                                        activation_scale,
                                        scale_bits[column],
                                        secondary_bits[column],
                                        emin_bits[column],
                                        IS_ASYMMETRIC
                                            ? (first_row
                                                   ? sum_A_row0
                                                   : sum_A_row1)
                                            : 0,
                                        (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                            ? (first_row
                                                   ? sum_A_lo_row0
                                                   : sum_A_lo_row1)
                                            : 0,
                                        (IS_DUAL_SCALE_ASYM || IS_IQ1_M)
                                            ? (first_row
                                                   ? sum_A_hi_row0
                                                   : sum_A_hi_row1)
                                            : 0,
                                        iq1m_qh[column],
                                        IS_IQ1_M ? subgroup_sums[0] : 0,
                                        IS_IQ1_M ? subgroup_sums[1] : 0,
                                        IS_IQ1_M ? subgroup_sums[2] : 0,
                                        IS_IQ1_M ? subgroup_sums[3] : 0);
                            acc[wi][wj][e] = __fadd_rn(
                                acc[wi][wj][e],
                                contribution);
                        }
                    }
                }
                commit_serial_m1_partition(kb + 1);
            }
        };

        // ── Dispatch: stream-K tile loop or standard single-tile ─────
        if constexpr (STREAM_K)
        {
            // Stream-K uses atomicAdd on pre-zeroed buffer — no need for beta/bias epilogue.
            (void)beta;
            (void)C_existing;
            (void)bias;
            if constexpr (!FIXUP_TWO_PASS)
                (void)tmp_fixup;

            // ── Stream-K tile loop (optimized) ──────────────────────────
            // Key optimizations vs naive stream-K:
            // 1. Full tiles (this CTA owns entire K-reduction) use direct store
            //    with interior fast path — matches standard epilogue exactly.
            // 2. Partial tiles (K-range split across CTAs) use atomicAdd on
            //    pre-zeroed C.
            // 3. Integer division (kbc → tile_idx, kt_begin) computed ONCE for
            //    the first tile; subsequent tiles use incremental tile_idx++.
            {
                const int ntx = (N + BN - 1) / BN;
                const int nty = (M + BM - 1) / BM;
                const int total_work = nty * ntx * num_k_tiles_total;
                int kbc = static_cast<int>(
                    (long long)blockIdx.x * total_work / gridDim.x);
                const int kbc_stop = static_cast<int>(
                    (long long)(blockIdx.x + 1) * total_work / gridDim.x);

                // First tile: compute tile index and K-offset via division (only once)
                int cur_tile_idx = kbc / num_k_tiles_total;
                int cur_kt_begin = kbc % num_k_tiles_total;

#pragma unroll 1
                while (kbc < kbc_stop)
                {
                    const int local_kt_end = min(num_k_tiles_total,
                                                 cur_kt_begin + (kbc_stop - kbc));
                    const int ty = cur_tile_idx / ntx;
                    const int tx = cur_tile_idx % ntx;
                    block_m = ty * BM;
                    block_n = tx * BN;
                    kt_begin = cur_kt_begin;
                    kt_end = local_kt_end;
                    num_k_iters = kt_end - kt_begin;
                    is_interior_tile_mut = (block_m + BM <= M) && (block_n + BN <= N);

                    // Full tile: this CTA owns the complete K-reduction (no sharing)
                    const bool is_full_tile = (cur_kt_begin == 0) &&
                                              (local_kt_end == num_k_tiles_total);

                    // ── Inline zero_acc ──
#pragma unroll
                    for (int i = 0; i < WM; ++i)
#pragma unroll
                        for (int j = 0; j < WN; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                            {
                                acc[i][j][e] = 0.0f;
                                serial_acc[i][j][e] = 0.0f;
                            }

                    // ── Inline k-loop (run_pipeline) ──
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

                    // ── Epilogue: direct store for full tiles, atomicAdd for partials ──
                    if constexpr (FIXUP_TWO_PASS)
                    {
                        const int ntx_local = (N + BN - 1) / BN;
                        const int tile_idx_f = (block_m / BM) * ntx_local + (block_n / BN);
                        float *__restrict__ tile_base = tmp_fixup + static_cast<long long>(tile_idx_f) * (BM * BN);
#pragma unroll
                        for (int wj = 0; wj < WN; ++wj)
                        {
                            const int local_n = wc * WARP_N + wj * 8;
                            const int lc0 = local_n + frag_col(lane_id, 0);
                            const int lc1 = local_n + frag_col(lane_id, 1);
#pragma unroll
                            for (int wi = 0; wi < WM; ++wi)
                            {
                                const int local_m = wr * WARP_M + wi * 16;
#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const int lr = local_m + frag_row(lane_id, e);
                                    const int lc = (e & 1) ? lc1 : lc0;
                                    atomicAdd(&tile_base[lr * BN + lc], acc[wi][wj][e] * alpha);
                                }
                            }
                        }
                    }
                    else if (is_full_tile)
                    {
                        // ── Full tile: direct store (matches standard epilogue) ──
                        // This CTA owns the complete K-reduction — no other CTA
                        // writes to these output elements. Use direct store with
                        // interior fast path (no bounds check, no atomicAdd).
#pragma unroll
                        for (int wj = 0; wj < WN; ++wj)
                        {
                            const int tile_n = block_n + wc * WARP_N + wj * 8;
                            const int gc0 = tile_n + frag_col(lane_id, 0);
                            const int gc1 = tile_n + frag_col(lane_id, 1);
#pragma unroll
                            for (int wi = 0; wi < WM; ++wi)
                            {
                                const int tile_m = block_m + wr * WARP_M + wi * 16;

                                if (is_interior_tile_mut)
                                {
                                    // Interior fast path: 4 direct stores, no bounds check
                                    const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                                    const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                                    const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                                    const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;
                                    C[out_idx0] = acc[wi][wj][0] * alpha;
                                    C[out_idx1] = acc[wi][wj][1] * alpha;
                                    C[out_idx2] = acc[wi][wj][2] * alpha;
                                    C[out_idx3] = acc[wi][wj][3] * alpha;
                                }
                                else
                                {
                                    // Border: direct store with bounds check
#pragma unroll
                                    for (int e = 0; e < 4; ++e)
                                    {
                                        const int gr = tile_m + frag_row(lane_id, e);
                                        const int gc = (e & 1) ? gc1 : gc0;
                                        if (gr < M && gc < N)
                                            C[gr * N + gc] = acc[wi][wj][e] * alpha;
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        // ── Partial tile: atomicAdd on pre-zeroed C ──
                        // K-range is split across CTAs; use atomicAdd for correctness.
#pragma unroll
                        for (int wj = 0; wj < WN; ++wj)
                        {
                            const int tile_n = block_n + wc * WARP_N + wj * 8;
                            const int gc0 = tile_n + frag_col(lane_id, 0);
                            const int gc1 = tile_n + frag_col(lane_id, 1);
#pragma unroll
                            for (int wi = 0; wi < WM; ++wi)
                            {
                                const int tile_m = block_m + wr * WARP_M + wi * 16;
#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const int gr = tile_m + frag_row(lane_id, e);
                                    const int gc = (e & 1) ? gc1 : gc0;
                                    if (gr < M && gc < N)
                                        atomicAdd(&C[gr * N + gc], acc[wi][wj][e] * alpha);
                                }
                            }
                        }
                    }

                    // Advance to next tile (no division needed — incremental tracking)
                    kbc += (num_k_tiles_total - cur_kt_begin);
                    cur_tile_idx++;
                    cur_kt_begin = 0; // all subsequent tiles start at K=0
                    __syncthreads();
                }
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
                    {
                        acc[i][j][e] = 0.0f;
                        serial_acc[i][j][e] = 0.0f;
                    }

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

            float output_alpha = alpha;
            if constexpr (SPLIT_K == 1)
            {
                if (serial_m1_uses_ordered_reducer)
                {
#pragma unroll
                    for (int i = 0; i < WM; ++i)
#pragma unroll
                        for (int j = 0; j < WN; ++j)
#pragma unroll
                            for (int e = 0; e < 4; ++e)
                                acc[i][j][e] = serial_acc[i][j][e];
                    output_alpha = 1.0f;
                }
            }

            // Epilogue: write accumulators to global memory
            const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

            // Two-phase split-K: each z-slice writes to partials at offset
            // blockIdx.z * M * N. The reduce kernel sums across z-slices.
            // For SPLIT_K == 1, C points to the final output directly.
            float *__restrict__ C_out = C;
            if constexpr (SPLIT_K > 1)
                C_out = C + blockIdx.z * M * N;

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

                    if (interior && (simple_epilogue || SPLIT_K > 1))
                    {
                        const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                        const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                        const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                        const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;

                        // Direct stores: for SPLIT_K > 1, beta/bias handled by reduce kernel
                        C_out[out_idx0] = acc[wi][wj][0] * output_alpha;
                        C_out[out_idx1] = acc[wi][wj][1] * output_alpha;
                        C_out[out_idx2] = acc[wi][wj][2] * output_alpha;
                        C_out[out_idx3] = acc[wi][wj][3] * output_alpha;
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
                            float val = acc[wi][wj][e] * output_alpha;

                            if constexpr (SPLIT_K == 1)
                            {
                                if (beta != 0.0f && C_existing)
                                    val += beta * C_existing[out_idx];
                                if (bias)
                                    val += (e & 1) ? bias1 : bias0;
                            }
                            // Direct store: for SPLIT_K > 1, beta/bias handled by reduce kernel
                            C_out[out_idx] = val;
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
        (void)serial_m1_k_partitions;
        (void)serial_m1_uses_ordered_reducer;
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
              bool ORDERED_M1 = false,
              int BLOCK_SIZE_ = WARPS_M * WARPS_N * 32,
              int MIN_BLOCKS_HINT = (BLOCK_SIZE_ >= 512 ? 1 : 2)>
    __global__ __launch_bounds__(BLOCK_SIZE_, MIN_BLOCKS_HINT) void nativeVnniTC_BK256(
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
        float beta,
        int serial_m1_k_partitions)
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
        static_assert(
            !ORDERED_M1 || SPLIT_K == 1,
            "Ordered public-M1 emulation owns the complete K reduction");

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
        float serial_acc[WM][WN][4];
#pragma unroll
        for (int i = 0; i < WM; ++i)
#pragma unroll
            for (int j = 0; j < WN; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    acc[i][j][e] = 0.0f;
                    if constexpr (ORDERED_M1)
                        serial_acc[i][j][e] = 0.0f;
                }

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
                if constexpr (!ORDERED_M1)
                {
                    /*
                     * Ordinary prefill preloads all four A fragments for this
                     * half. That maximizes instruction-level parallelism when
                     * no second reducer accumulator is live.
                     */
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
                            const int a_row_base =
                                wr * WARP_M + wi * 16;
                            load_ldmatrix_a_m16n8k32(
                                A_frag_all[wi][k],
                                reinterpret_cast<const int *>(
                                    &smem_A[
                                        a_row_base * BK128_STRIDE +
                                        k_offset]),
                                BK128_STRIDE / 4,
                                lane_id);

                            const int local_row0 =
                                a_row_base + gid;
                            sa_all[wi][0][k] =
                                smem_sa[
                                    local_row0 * 8 + scale_idx];
                            sa_all[wi][1][k] =
                                smem_sa[
                                    (local_row0 + 8) * 8 +
                                    scale_idx];
                        }
                    }

#pragma unroll
                    for (int k = 0; k < 4; ++k)
                    {
                        const int b_k_offset =
                            hi * 128 + k * 32;
                        const int scale_idx_b = hi * 4 + k;

                        uint16_t scale_bits_pre[WN][2];
#pragma unroll
                        for (int wj = 0; wj < WN; ++wj)
                        {
                            const int b_col_base =
                                wc * WARP_N + wj * 8;
                            scale_bits_pre[wj][0] =
                                smem_scales_B[
                                    scale_idx_b * BN +
                                    b_col_base +
                                    frag_col(lane_id, 0)];
                            scale_bits_pre[wj][1] =
                                smem_scales_B[
                                    scale_idx_b * BN +
                                    b_col_base +
                                    frag_col(lane_id, 1)];
                        }

#pragma unroll
                        for (int wj = 0; wj < WN; ++wj)
                        {
                            uint32_t B_frag[2];
                            load_ldmatrix_b_m16n8k32(
                                B_frag,
                                reinterpret_cast<const int *>(
                                    &smem_B[
                                        (wc * WARP_N + wj * 8) *
                                            BK256_STRIDE +
                                        b_k_offset]),
                                BK256_STRIDE / 4,
                                lane_id);

#pragma unroll
                            for (int wi = 0; wi < WM; ++wi)
                            {
                                int32_t D[4] = {0, 0, 0, 0};
                                mma_m16n8k32_s8(
                                    D,
                                    A_frag_all[wi][k],
                                    B_frag);

#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const int column = e & 1;
                                    const float activation_scale =
                                        (e < 2)
                                            ? sa_all[wi][0][k]
                                            : sa_all[wi][1][k];
                                    const float contribution =
                                        llaminar2::
                                            cuda_native_vnni::
                                                native_vnni_block_contribution_from_reduced_terms_rn<
                                                    0>(
                                                    D[e],
                                                    0,
                                                    activation_scale,
                                                    scale_bits_pre[wj][column],
                                                    uint16_t{0},
                                                    uint32_t{0},
                                                    0,
                                                    0,
                                                    0,
                                                    uint16_t{0},
                                                    0,
                                                    0,
                                                    0,
                                                    0);
                                    acc[wi][wj][e] =
                                        __fadd_rn(
                                            acc[wi][wj][e],
                                            contribution);
                                }
                            }
                        }
                    }
                }
                else
                {
                    /*
                     * Ordered M1 equivalence keeps both the current partition
                     * and ascending reducer totals live. Load only one
                     * 32-value A fragment per row at a time, reducing fragment
                     * liveness from 32 registers to eight for this tile
                     * geometry. B still loads once per (K block, output
                     * fragment) and is reused by every local M fragment.
                     */
                    const int blocks_per_partition =
                        (num_q40_blocks +
                         serial_m1_k_partitions - 1) /
                        serial_m1_k_partitions;
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                    {
                        const int completed_block =
                            ot * 8 + hi * 4 + k + 1;
                        if (completed_block > num_q40_blocks)
                            break;

                        const int k_offset = k * 32;
                        const int b_k_offset =
                            hi * 128 + k * 32;
                        const int scale_idx = hi * 4 + k;

                        uint32_t A_frag[WM][4];
                        float sa[WM][2];
#pragma unroll
                        for (int wi = 0; wi < WM; ++wi)
                        {
                            const int a_row_base =
                                wr * WARP_M + wi * 16;
                            load_ldmatrix_a_m16n8k32(
                                A_frag[wi],
                                reinterpret_cast<const int *>(
                                    &smem_A[
                                        a_row_base * BK128_STRIDE +
                                        k_offset]),
                                BK128_STRIDE / 4,
                                lane_id);
                            const int local_row0 =
                                a_row_base + gid;
                            sa[wi][0] =
                                smem_sa[
                                    local_row0 * 8 + scale_idx];
                            sa[wi][1] =
                                smem_sa[
                                    (local_row0 + 8) * 8 +
                                    scale_idx];
                        }

#pragma unroll
                        for (int wj = 0; wj < WN; ++wj)
                        {
                            const int b_col_base =
                                wc * WARP_N + wj * 8;
                            const uint16_t scale_bits[2] = {
                                smem_scales_B[
                                    scale_idx * BN +
                                    b_col_base +
                                    frag_col(lane_id, 0)],
                                smem_scales_B[
                                    scale_idx * BN +
                                    b_col_base +
                                    frag_col(lane_id, 1)]};
                            uint32_t B_frag[2];
                            load_ldmatrix_b_m16n8k32(
                                B_frag,
                                reinterpret_cast<const int *>(
                                    &smem_B[
                                        b_col_base *
                                            BK256_STRIDE +
                                        b_k_offset]),
                                BK256_STRIDE / 4,
                                lane_id);

#pragma unroll
                            for (int wi = 0; wi < WM; ++wi)
                            {
                                int32_t D[4] = {0, 0, 0, 0};
                                mma_m16n8k32_s8(
                                    D,
                                    A_frag[wi],
                                    B_frag);
#pragma unroll
                                for (int e = 0; e < 4; ++e)
                                {
                                    const int column = e & 1;
                                    const float contribution =
                                        llaminar2::
                                            cuda_native_vnni::
                                                native_vnni_block_contribution_from_reduced_terms_rn<
                                                    0>(
                                                    D[e],
                                                    0,
                                                    (e < 2)
                                                        ? sa[wi][0]
                                                        : sa[wi][1],
                                                    scale_bits[column],
                                                    uint16_t{0},
                                                    uint32_t{0},
                                                    0,
                                                    0,
                                                    0,
                                                    uint16_t{0},
                                                    0,
                                                    0,
                                                    0,
                                                    0);
                                    acc[wi][wj][e] =
                                        __fadd_rn(
                                            acc[wi][wj][e],
                                            contribution);
                                }
                            }
                        }

                        const bool partition_complete =
                            completed_block ==
                                num_q40_blocks ||
                            (completed_block %
                             blocks_per_partition) == 0;
                        if (partition_complete)
                        {
#pragma unroll
                            for (int wi = 0; wi < WM; ++wi)
#pragma unroll
                                for (int wj = 0;
                                     wj < WN;
                                     ++wj)
#pragma unroll
                                    for (int e = 0;
                                         e < 4;
                                         ++e)
                                    {
                                        const float partial =
                                            __fmul_rn(
                                                alpha,
                                                acc[wi][wj][e]);
                                        serial_acc[wi][wj][e] =
                                            __fadd_rn(
                                                serial_acc[wi][wj][e],
                                                partial);
                                        acc[wi][wj][e] = 0.0f;
                                    }
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
        if constexpr (ORDERED_M1)
        {
#pragma unroll
            for (int wi = 0; wi < WM; ++wi)
#pragma unroll
                for (int wj = 0; wj < WN; ++wj)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        acc[wi][wj][e] =
                            serial_acc[wi][wj][e];
        }
        const float output_alpha =
            ORDERED_M1 ? 1.0f : alpha;
        const bool simple_epilogue = (beta == 0.0f) && (bias == nullptr);

        // Two-phase split-K: each z-slice writes to partials at offset
        // blockIdx.z * M * N. The reduce kernel sums across z-slices.
        float *__restrict__ C_out = C;
        if constexpr (SPLIT_K > 1)
            C_out = C + blockIdx.z * M * N;

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

                if (interior && (simple_epilogue || SPLIT_K > 1))
                {
                    const int out_idx0 = (tile_m + frag_row(lane_id, 0)) * N + gc0;
                    const int out_idx1 = (tile_m + frag_row(lane_id, 1)) * N + gc1;
                    const int out_idx2 = (tile_m + frag_row(lane_id, 2)) * N + gc0;
                    const int out_idx3 = (tile_m + frag_row(lane_id, 3)) * N + gc1;

                    C_out[out_idx0] =
                        __fmul_rn(acc[wi][wj][0], output_alpha);
                    C_out[out_idx1] =
                        __fmul_rn(acc[wi][wj][1], output_alpha);
                    C_out[out_idx2] =
                        __fmul_rn(acc[wi][wj][2], output_alpha);
                    C_out[out_idx3] =
                        __fmul_rn(acc[wi][wj][3], output_alpha);
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
                        float val =
                            __fmul_rn(acc[wi][wj][e], output_alpha);

                        if constexpr (SPLIT_K == 1)
                        {
                            if (beta != 0.0f && C_existing)
                                val += beta * C_existing[out_idx];
                            if (bias)
                                val += (e & 1) ? bias1 : bias0;
                        }
                        C_out[out_idx] = val;
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
        (void)serial_m1_k_partitions;
#endif
    }

    // =========================================================================
    // Tile/split-k force: override the heuristic for sweep benchmarks.
    //   g_force_tile_id: -1 = auto (heuristic), 0..5 = TileId enum value
    //   g_force_split_k:  0 = auto (heuristic), 1..8 = forced split-K value
    // =========================================================================
    static int g_force_tile_id = []()
    {
        return llaminar2::debugEnv().gemm.cuda_force_prefill_tile;
    }();
    static int g_force_split_k = []()
    {
        return llaminar2::debugEnv().gemm.cuda_force_prefill_split_k;
    }();

    // Deterministic mode remains a diagnostic control for explicitly forced
    // candidates. Production auto dispatch is batch invariant unconditionally:
    // it never needs this switch to reject atomic or differently parenthesized
    // K reductions.
    static bool g_deterministic_mode = []()
    {
        return llaminar2::debugEnv().gemm.deterministic;
    }();

    // BK256 mode: 0=auto (heuristic), 1=force ON, -1=force OFF
    // Set via LLAMINAR_BK256_MODE env var or extern C API.
    static int g_bk256_force_mode = []()
    {
        return llaminar2::debugEnv().gemm.cuda_bk256_mode;
    }();

    // ─── Format complexity classification ─────────────────────────────
    // Format complexity used by the total ordinary-prefill heuristic. Learned
    // dispatch is intentionally limited to M=1 and grouped verifier rows.
    enum class FormatComplexity
    {
        Simple,     // CB 0,4,6,11,12,15,19 – single-scale, Q4_0 heuristic works
        Asymmetric, // CB 5,7,16 – min-correction overhead, needs w2x2 bias
        DualScale   // CB 8,9,10,13,14,17 – dual-scale (profitability-gated)
    };

    constexpr FormatComplexity getFormatComplexity(uint8_t cb)
    {
        switch (cb)
        {
        case 5:
        case 7:
        case 16:
            return FormatComplexity::Asymmetric;
        case 8:
        case 9:
        case 10:
        case 13:
        case 14:
        case 17:
            return FormatComplexity::DualScale;
        default:
            return FormatComplexity::Simple;
        }
    }

    // ─── Asymmetric-format heuristic ──────────────────────────────────
    // Broad sweep data (Q4_1, Q5_1, IQ1_S across 7B shapes, M=64..512)
    // retains w2x2 as the conservative general rule. Exact production
    // geometries may override it only after a byte-gated tile tournament and
    // an isolated occupancy/spill profile establish a better winner.
    TileChoice choosePrefillTile_Asymmetric(
        int M,
        int N,
        int K,
        uint8_t codebook,
        CUDAPrefillContext_ *prefill_ctx)
    {
        const int SM = querySmCount(prefill_ctx);
        const int t64x128 = ((M + 63) / 64) * ((N + 127) / 128);

        /*
         * Qwen3.6 MoE GDN QKV projection, 128-row graph bucket.
         *
         * A 50-sample production-path tournament proved every candidate
         * byte-identical to the exact-M AUTO oracle. Q4_1 and Q5_1 selected
         * T64x128_w2x4; IQ1_S selected T64x128_w4x2. On GA102, profiling the
         * Q4_1 winner reduced registers from 168 to 128 per thread, doubled
         * theoretical occupancy from 16.67% to 33.33%, raised achieved
         * occupancy from 13.34% to 26.42%, and retained zero local-memory
         * spills. Keep this overlay exact instead of extrapolating one measured
         * geometry over the older all-shape heuristic.
         */
        if (M == 128 && N == 8192 && K == 2048)
        {
            if (codebook == 5 || codebook == 7)
                return {TileId::T64x128_w2x4, 1};
            if (codebook == 16)
                return {TileId::T64x128_w4x2, 1};
        }

        // Well-filling: enough tiles to saturate SMs → w2x2, no split-K
        if (t64x128 >= SM)
            return {TileId::T64x128_w2x2, 1};

        /*
         * Underfilled shapes still retain one ordered K walk per output row.
         * Split-K partials cannot reconstruct the serial FP32 addition sequence:
         * reducing two rounded partition sums changes parenthesization. Recover
         * occupancy with output geometry, not a mathematically different K tree.
         */
        return {TileId::T64x128_w2x2, 1};
    }

    // ─── Sweep-derived tile + split_k heuristic ───────────────────────
    // Fills-first strategy: prefer the largest output tile family that fills the
    // GPU while preserving one canonical, increasing-K reduction per row.
    // Within 64×128, warp config depends on real tile count:
    //   tiles ≥ 112 → w2x2 (more blocks/SM at high tile count)
    //   ki ≤ 7      → w4x2 (small K, 4 warps in M-dim)
    //   otherwise   → w2x4 (default, 8 warps for ILP)
    TileChoice choosePrefillTile(int M, int N, int K,
                                 CUDAPrefillContext_ *prefill_ctx,
                                 FormatComplexity complexity = FormatComplexity::Simple,
                                 uint8_t codebook = 0)
    {
        // Force-tile override for sweep benchmarks
        if (g_force_tile_id >= 0 && g_force_tile_id <= 5)
        {
            const int sk = (g_force_split_k > 0) ? g_force_split_k : 1;
            return {static_cast<TileId>(g_force_tile_id), sk};
        }

        // Asymmetric/dual-scale formats: specialized heuristic biased
        // toward w2x2 due to higher register pressure from min-correction
        if (complexity != FormatComplexity::Simple)
            return choosePrefillTile_Asymmetric(
                M, N, K, codebook, prefill_ctx);

        const int SM = querySmCount(prefill_ctx);
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
            // w2x2 (4 warps/block) for very high tile counts where more
            // blocks improve wave fill; 2*SM threshold from sweep data
            // across 0.5B/3B/7B/14B shapes.
            if (real_tiles >= 2 * SM)
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
            // Short K-loop (ki 8-16) + abundant 64×128 tiles → T64x128 has
            // better SM utilization. T128x128 register overhead can't be
            // amortized with few K-iterations. Sweep: 3B_FFN_Up M=128,256.
            else if (ki > 7 && ki <= 16 && t64x128 >= 2 * SM)
                use_128 = false;
            // Marginal T128 wave fill (<1.5 waves) + T64x128 fills well (≥2 waves)
            // → demote. T64x128's higher occupancy wins over T128's larger
            // tile when T128 can't fill a second wave efficiently.
            // Sweep: 7B_Attn M=512 (ki=28), 7B_FFN_Down M=512 (ki=148).
            // Safe: 7B_FFN_Up M=128 has t128=148 > 3*SM/2, keeps T128.
            else if (ki > 16 && t128 < 3 * SM / 2 && t64x128 >= 2 * SM)
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

            const TileId warp = pick_warp(t64x128, ki);

            // 128×128 override for very large K at moderate M.
            if (M >= 256 && ki >= 40 && t128 >= 32 && t64x128 <= (3 * SM / 2))
                return {TileId::T128x128_w4x2, 1};

            // Prefer the larger tile when its output grid remains economical.
            if (M >= 128 && ki >= 40 && t128 >= 32 && t64x128 >= (3 * SM / 2))
                return {TileId::T128x128_w4x2, 1};

            return {warp, 1};
        }

        // ═══ TIER 3: 64×64 fills ═══
        if (fills_64)
        {
            // A wider output tile can amortize large-K setup without changing
            // the K reduction tree.
            if (ki >= 14 && t64x128 >= 28)
            {
                const TileId warp = pick_warp(t64x128, ki);

                // Further upgrade for very large K when a useful output grid
                // remains available.
                if (t128 >= 16 && ki >= 40 && M >= 128)
                    return {TileId::T128x128_w4x2, 1};

                return {warp, 1};
            }
            return {TileId::T64x64_w2x2, 1};
        }

        // ═══ TIER 4: Nothing fills → finest useful output tile ═══
        TileId tile;
        if (t128 >= 16 && ki >= 40 && M >= 128)
        {
            tile = TileId::T128x128_w4x2;
        }
        else if (t64x128 >= 8 && ki >= 8)
        {
            tile = pick_warp(0, ki); // will yield w2x4 or w4x2
        }
        else
        {
            tile = TileId::T64x64_w2x2;
        }
        return {tile, 1};
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

    template <uint8_t CODEBOOK_ID, int BM, int BN, int WM, int WN, int SPLIT_K = 1>
    bool launchNativeVNNITC_BK64(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const int32_t *d_sums_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream,
        CUDAPrefillContext_ *prefill_ctx = nullptr,
        int serial_m1_k_partitions = 1,
        int serial_m1_uses_ordered_reducer = 0)
    {
        const int num_k_tiles = (K / 32 + 1) / 2; // ceil: handles K%64!=0
        int kt_per_part = num_k_tiles;
        if constexpr (SPLIT_K > 1)
            kt_per_part = (num_k_tiles + SPLIT_K - 1) / SPLIT_K;
        if (kt_per_part <= 0)
            return false;

        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        // Two-phase split-K: allocate partials, pass as C, reduce after
        float *d_kernel_C = d_C_fp32;
        if constexpr (SPLIT_K > 1)
        {
            const size_t partials_bytes = static_cast<size_t>(SPLIT_K) * M * N * sizeof(float);
            float *partials = prefill_ctx ? getOrAllocSplitkPartials(prefill_ctx, partials_bytes, cuda_stream) : nullptr;
            if (!partials)
                return false;
            d_kernel_C = partials;
        }

        // Clear any stale CUDA error from prior operations (e.g. CUTLASS reference path)
        (void)cudaGetLastError();

        nativeVnniTC_BK64<CODEBOOK_ID, BM, BN, WM, WN, SPLIT_K><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_mins,
            d_emins,
            d_kernel_C,
            d_scales_A_block,
            d_sums_A_block,
            d_C_existing,
            d_bias,
            M,
            N,
            K,
            alpha,
            beta,
            serial_m1_k_partitions,
            serial_m1_uses_ordered_reducer,
            nullptr);
        if (cudaGetLastError() != cudaSuccess)
            return false;

        // Launch reduce kernel to sum partials into final output
        if constexpr (SPLIT_K > 1)
        {
            const int total = M * N;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            splitk_reduce<<<blocks, threads, 0, cuda_stream>>>(
                d_kernel_C, d_C_fp32, d_C_existing, d_bias,
                M, N, SPLIT_K, beta);
            if (cudaGetLastError() != cudaSuccess)
                return false;
        }
        return true;
    }

    // =========================================================================
    // Stream-K diagnostic mode: 0/-1 = disabled, 1 = force one-pass,
    // 2 = force two-pass. Atomic Stream-K is intentionally absent from automatic
    // production dispatch because its scheduler-dependent reduction cannot meet
    // the batch-invariant publication contract.
    // =========================================================================
    static int g_stream_k_force_mode = []()
    {
        return llaminar2::debugEnv().gemm.cuda_stream_k_mode;
    }();

    // =========================================================================
    // Bias-add kernel: adds per-column bias to C (launched after stream-K GEMM)
    // C[row, col] += bias[col] for all (row, col) in [M, N].
    // =========================================================================
    __global__ void streamk_add_bias(
        float *__restrict__ C,
        const float *__restrict__ bias,
        int M,
        int N)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = M * N;
        if (idx < total)
            C[idx] += bias[idx % N];
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
        const int32_t *d_sums_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream,
        CUDAPrefillContext_ *prefill_ctx)
    {
        const int nsm = querySmCount(prefill_ctx);

        // Query how many blocks the hardware can run concurrently per SM,
        // given the kernel's compiled register and shared memory usage.
        // This lets us fill more SM slots when registers permit (e.g., 128-thread
        // tiles typically fit 2 blocks/SM, doubling occupancy vs 1 block/SM).
        constexpr int BLOCK_SIZE_SK = WM * WN * 32;
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            nativeVnniTC_BK64<CODEBOOK_ID, BM, BN, WM, WN, 1, 2, true>,
            BLOCK_SIZE_SK, 0);
        // Cap at 2 to limit work fragmentation; at least 1
        const int occ_mult = max(1, min(max_blocks_per_sm, 2));
        const int grid_blocks = occ_mult * nsm;

        (void)cudaGetLastError();

        // Stream-K with atomicAdd requires C to be zeroed so that all blocks
        // can safely atomicAdd their partial sums.  Beta is not supported
        // because there is no ordering guarantee on which block writes first.
        // Bias is handled as a post-hoc add after the GEMM completes.
        if (beta != 0.0f)
            return false;

        cudaMemsetAsync(d_C_fp32, 0, static_cast<size_t>(M) * N * sizeof(float), cuda_stream);

        // Main kernel: grid_blocks persistent blocks. Partial tiles use
        // atomicAdd directly on C (no fixup buffer or fixup kernel needed).
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
            d_sums_A_block,
            nullptr, // d_C_existing: not supported with stream-K
            nullptr, // d_bias: applied post-hoc below
            M, N, K,
            alpha, 0.0f,
            1, 0,
            nullptr); // No fixup buffer needed — partials use atomicAdd on C

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "[StreamK] Kernel launch failed: %s (M=%d N=%d K=%d grid=%d occ_mult=%d)\n",
                    cudaGetErrorString(err), M, N, K, grid_blocks, occ_mult);
            return false;
        }

        // Post-hoc bias add: safe because GEMM writes are complete before
        // this kernel reads C (CUDA stream ordering guarantees).
        if (d_bias != nullptr)
        {
            const int total_elems = M * N;
            constexpr int BIAS_THREADS = 256;
            const int bias_blocks = (total_elems + BIAS_THREADS - 1) / BIAS_THREADS;
            streamk_add_bias<<<bias_blocks, BIAS_THREADS, 0, cuda_stream>>>(
                d_C_fp32, d_bias, M, N);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "[StreamK] Bias-add kernel failed: %s\n", cudaGetErrorString(err));
                return false;
            }
        }

        return true;
    }

    // =========================================================================
    // Stream-K two-pass fixup kernel: copies tile-indexed fixup buffer to C
    // with M/N bounds checking. One thread block per output tile.
    // =========================================================================
    template <int BM, int BN>
    __global__ void streamk_fixup_copy(
        const float *__restrict__ fixup,
        float *__restrict__ C,
        int M,
        int N,
        int ntx)
    {
        const int tile_idx = blockIdx.x;
        const int ty = tile_idx / ntx;
        const int tx = tile_idx % ntx;
        const int block_m = ty * BM;
        const int block_n = tx * BN;

        const float *__restrict__ tile_data = fixup + tile_idx * (BM * BN);

        // Each thread handles multiple elements via grid-stride within the tile
        for (int i = threadIdx.x; i < BM * BN; i += blockDim.x)
        {
            const int lr = i / BN; // BN is power of 2 → compiler uses shift
            const int lc = i % BN; // BN is power of 2 → compiler uses mask
            const int gr = block_m + lr;
            const int gc = block_n + lc;
            if (gr < M && gc < N)
                C[gr * N + gc] = tile_data[i];
        }
    }

    // =========================================================================
    // Stream-K two-pass launch helper: main kernel writes partial sums to a
    // flat tile-indexed fixup buffer (atomicAdd, no M/N bounds check), then a
    // lightweight fixup kernel copies fixup→C with bounds checking.
    //
    // The main kernel uses MIN_BLOCKS_HINT=2, targeting ≤64 regs/thread for
    // 2 blocks/SM (66.67% occupancy) on 512-thread tiles. The simpler epilogue
    // (no runtime N multiply, no M/N bounds checks) eliminates ~12 gap
    // registers that forced the one-pass variant to 128 regs / 1 block/SM.
    // =========================================================================
    template <uint8_t CODEBOOK_ID, int BM, int BN, int WM, int WN>
    bool launchNativeVNNITC_BK64_StreamK_TwoPass(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const int32_t *d_sums_A_block,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        cudaStream_t cuda_stream,
        CUDAPrefillContext_ *prefill_ctx)
    {
        // Two-pass stream-K only supports beta=0, no bias (same constraint as one-pass)
        if (beta != 0.0f || d_bias != nullptr)
            return false;

        const int nsm = querySmCount(prefill_ctx);
        const int ntx = (N + BN - 1) / BN;
        const int nty = (M + BM - 1) / BM;
        const int total_tiles = ntx * nty;

        // Query occupancy for the two-pass kernel (FIXUP_TWO_PASS=true → MIN_BLOCKS_HINT=2)
        constexpr int BLOCK_SIZE_SK = WM * WN * 32;
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm,
            nativeVnniTC_BK64<CODEBOOK_ID, BM, BN, WM, WN, 1, 2, /*STREAM_K=*/true, /*FIXUP_TWO_PASS=*/true>,
            BLOCK_SIZE_SK, 0);
        const int occ_mult = max(1, min(max_blocks_per_sm, 2));
        const int grid_blocks = occ_mult * nsm;

        (void)cudaGetLastError();

        // Allocate tile-indexed fixup buffer: total_tiles × BM × BN floats
        const size_t fixup_bytes = static_cast<size_t>(total_tiles) * BM * BN * sizeof(float);
        float *fixup_buf = getOrAllocFixupBuffer(prefill_ctx, fixup_bytes, cuda_stream);
        if (!fixup_buf)
            return false;

        // Main kernel: writes partial sums to fixup buffer via atomicAdd.
        // No M/N bounds checks — fixup buffer covers full padded tiles.
        const dim3 grid(grid_blocks, 1, 1);
        const dim3 block(BLOCK_SIZE_SK);

        nativeVnniTC_BK64<CODEBOOK_ID, BM, BN, WM, WN, /*SPLIT_K=*/1, /*STAGES_=*/2,
                          /*STREAM_K=*/true, /*FIXUP_TWO_PASS=*/true><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_mins,
            d_emins,
            d_C_fp32,
            d_scales_A_block,
            d_sums_A_block,
            d_C_existing,
            d_bias,
            M, N, K,
            alpha, beta,
            1, 0,
            fixup_buf);

        if (cudaGetLastError() != cudaSuccess)
            return false;

        // Fixup kernel: copy from tile-indexed fixup buffer to C with M/N bounds checking.
        // One thread block per output tile, 256 threads each.
        constexpr int FIXUP_THREADS = 256;
        streamk_fixup_copy<BM, BN><<<total_tiles, FIXUP_THREADS, 0, cuda_stream>>>(
            fixup_buf, d_C_fp32, M, N, ntx);

        return cudaGetLastError() == cudaSuccess;
    }

    // =========================================================================
    // Stream-K eligibility is explicit and diagnostic only. Automatic production
    // dispatch must preserve the canonical increasing-K FP32 reduction.
    // =========================================================================
    bool shouldUseStreamK(int M, int N, int K, int bm, int bn, CUDAPrefillContext_ *prefill_ctx)
    {
        (void)M;
        (void)N;
        (void)K;
        (void)bm;
        (void)bn;
        (void)prefill_ctx;

        // Deterministic mode rejects even an explicitly requested diagnostic.
        if (g_deterministic_mode)
            return false;

        return g_stream_k_force_mode > 0;
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
        cudaStream_t cuda_stream,
        CUDAPrefillContext_ *prefill_ctx,
        int serial_m1_k_partitions,
        int serial_m1_uses_ordered_reducer)
    {
        // Compute dynamic smem size (must match kernel layout: A uses K=128 half)
        constexpr int SCALES_B_OFF = BM * BK128_STRIDE + BN * BK256_STRIDE;
        constexpr int SA_OFF = SCALES_B_OFF + 8 * BN * static_cast<int>(sizeof(uint16_t));
        constexpr int SA_ALIGNED = (SA_OFF + 3) & ~3;
        constexpr int smem_bytes = SA_ALIGNED + BM * 8 * static_cast<int>(sizeof(float));

        if constexpr (SPLIT_K > 1)
        {
            if (serial_m1_uses_ordered_reducer)
                return false;
        }

        /*
         * Ordinary and ordered-M1 kernels have intentionally different
         * register schedules, so configure and launch the exact specialization
         * selected by the immutable public-M1 arithmetic contract.
         */
        static bool smem_configured[2] = {false, false};
        const int ordered_index =
            serial_m1_uses_ordered_reducer ? 1 : 0;
        if (!smem_configured[ordered_index])
        {
            cudaError_t attribute_status = cudaSuccess;
            if (serial_m1_uses_ordered_reducer)
            {
                auto fn_ptr =
                    nativeVnniTC_BK256<
                        BM,
                        BN,
                        WM,
                        WN,
                        SPLIT_K,
                        true>;
                attribute_status = cudaFuncSetAttribute(
                    fn_ptr,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_bytes);
            }
            else
            {
                auto fn_ptr =
                    nativeVnniTC_BK256<
                        BM,
                        BN,
                        WM,
                        WN,
                        SPLIT_K,
                        false>;
                attribute_status = cudaFuncSetAttribute(
                    fn_ptr,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_bytes);
            }
            if (attribute_status != cudaSuccess)
                return false;
            smem_configured[ordered_index] = true;
        }

        const int num_outer_tiles = (K / 32 + 7) / 8;
        int ot_per_part = num_outer_tiles;
        if constexpr (SPLIT_K > 1)
            ot_per_part = (num_outer_tiles + SPLIT_K - 1) / SPLIT_K;
        if (ot_per_part <= 0)
            return false;

        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, SPLIT_K);
        const dim3 block(WM * WN * 32);

        // Two-phase split-K: allocate partials, pass as C, reduce after
        float *d_kernel_C = d_C_fp32;
        if constexpr (SPLIT_K > 1)
        {
            const size_t partials_bytes = static_cast<size_t>(SPLIT_K) * M * N * sizeof(float);
            float *partials = prefill_ctx ? getOrAllocSplitkPartials(prefill_ctx, partials_bytes, cuda_stream) : nullptr;
            if (!partials)
                return false;
            d_kernel_C = partials;
        }

        (void)cudaGetLastError(); // clear stale errors

        if (serial_m1_uses_ordered_reducer)
        {
            nativeVnniTC_BK256<
                BM,
                BN,
                WM,
                WN,
                SPLIT_K,
                true><<<grid, block, smem_bytes, cuda_stream>>>(
                d_A_int8,
                d_payload,
                d_scales,
                d_kernel_C,
                d_scales_A_block,
                d_C_existing,
                d_bias,
                M,
                N,
                K,
                alpha,
                beta,
                serial_m1_k_partitions);
        }
        else
        {
            nativeVnniTC_BK256<
                BM,
                BN,
                WM,
                WN,
                SPLIT_K,
                false><<<grid, block, smem_bytes, cuda_stream>>>(
                d_A_int8,
                d_payload,
                d_scales,
                d_kernel_C,
                d_scales_A_block,
                d_C_existing,
                d_bias,
                M,
                N,
                K,
                alpha,
                beta,
                serial_m1_k_partitions);
        }
        if (cudaGetLastError() != cudaSuccess)
            return false;

        if constexpr (SPLIT_K > 1)
        {
            const int total = M * N;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            splitk_reduce<<<blocks, threads, 0, cuda_stream>>>(
                d_kernel_C, d_C_fp32, d_C_existing, d_bias,
                M, N, SPLIT_K, beta);
            if (cudaGetLastError() != cudaSuccess)
                return false;
        }
        return true;
    }

    struct PrefillWorkspacePlan
    {
        int tile_id = -1;
        int split_k = 1;
        int streamk = 0;
        bool bk256 = false;
        size_t splitk_partials_bytes = 0;
        size_t streamk_fixup_bytes = 0;
    };

    /**
     * @brief Complete Q4_0 large-K route shared by planning and execution.
     *
     * Workspace planning and kernel launch must consume the same immutable
     * decision. Duplicating these predicates previously allowed planning to
     * reserve a T64x64 route while execution silently launched BK256. Keeping
     * the family typed also makes every supported route visible at each
     * exhaustive switch.
     */
    enum class Q40PrefillRoute
    {
        Generic,
        BK256Narrow,
        BK256Wide,
        ProfiledT64x64,
    };

    /**
     * @brief Select one Q4_0 large-K route without allocating or launching.
     *
     * An explicit BK256 force is strongest. An explicit ordinary tile force
     * then disables every AUTO overlay so tournament candidates remain
     * forceable. Production AUTO uses BK256 for its profitable first 32 rows,
     * the profiled T64x64 crossover for narrow larger-M work, and otherwise the
     * total generic tile heuristic.
     */
    Q40PrefillRoute chooseQ40PrefillRoute(
        int M,
        int N,
        int K,
        CUDAPrefillContext_ *prefill_ctx)
    {
        const auto bk256_geometry = [&]()
        {
            return (M <= 32) && (N <= 1024)
                ? Q40PrefillRoute::BK256Narrow
                : Q40PrefillRoute::BK256Wide;
        };

        if (g_bk256_force_mode > 0)
            return bk256_geometry();
        if (g_force_tile_id >= 0 && g_force_tile_id <= 5)
            return Q40PrefillRoute::Generic;

        /*
         * BK256 wins while a narrow projection fits in its first 32-row tile.
         * Above that boundary, the second BK256 M fragment increases latency
         * without increasing the four-to-eight block output grid. Production
         * tournaments on the 512x2048, 896x4864, and 1024x5120 Qwen
         * geometries consistently selected T64x64 from M=33 onward. Its finer
         * output grid was 16-34% faster on GA102, while the all-format
         * equivalence sweep proved the same increasing-K arithmetic byte for
         * byte.
         */
        if ((M > 32) && (N <= 1024) && (K > 2 * N))
            return Q40PrefillRoute::ProfiledT64x64;

        const bool bk256_disabled = (g_bk256_force_mode < 0);
        const int SM = querySmCount(prefill_ctx);
        const int t64x128 = ((M + 63) / 64) * ((N + 127) / 128);
        if (!bk256_disabled && (K > 2 * N) && (t64x128 < SM))
            return bk256_geometry();
        return Q40PrefillRoute::Generic;
    }

    void tileShape(TileId tile, int &bm, int &bn)
    {
        switch (tile)
        {
        case TileId::T64x64_w2x2:
            bm = 64;
            bn = 64;
            return;
        case TileId::T64x128_w2x2:
        case TileId::T64x128_w4x2:
        case TileId::T64x128_w2x4:
            bm = 64;
            bn = 128;
            return;
        case TileId::T128x128_w4x2:
        case TileId::T128x128_w4x4:
            bm = 128;
            bn = 128;
            return;
        }
        bm = 64;
        bn = 128;
    }

    template <uint8_t CB>
    PrefillWorkspacePlan planGenericPrefillWorkspace(
        int M,
        int N,
        int K,
        CUDAPrefillContext_ *prefill_ctx)
    {
        PrefillWorkspacePlan plan;

        if constexpr (CB == 0)
        {
            const Q40PrefillRoute route =
                chooseQ40PrefillRoute(M, N, K, prefill_ctx);
            switch (route)
            {
            case Q40PrefillRoute::BK256Narrow:
            case Q40PrefillRoute::BK256Wide:
            {
                /*
                 * BK256 retains one ordered K walk. A two-part workspace
                 * reduction is deterministic in scheduling but not equivalent
                 * in FP32 parenthesization, so it is not a production candidate.
                 */
                constexpr int sk = 1;
                plan.tile_id =
                    route == Q40PrefillRoute::BK256Narrow ? -3 : -2;
                plan.split_k = sk;
                plan.bk256 = true;
                return plan;
            }
            case Q40PrefillRoute::ProfiledT64x64:
            {
                /*
                 * Lock the measured winner directly instead of passing through
                 * the older generic heuristic, whose eight-block T64x128
                 * choice remains underfilled at the N=1024 boundary.
                 */
                plan.tile_id =
                    static_cast<int>(TileId::T64x64_w2x2);
                plan.split_k = 1;
                return plan;
            }
            case Q40PrefillRoute::Generic:
                break;
            }
        }

        TileChoice tc;
        if (g_force_tile_id >= 0 && g_force_tile_id <= 5)
        {
            tc = {static_cast<TileId>(g_force_tile_id),
                  (g_force_split_k > 0) ? g_force_split_k : 1};
        }
        else
        {
            constexpr FormatComplexity complexity = getFormatComplexity(CB);
            tc = choosePrefillTile(
                M, N, K, prefill_ctx, complexity, CB);
        }

        if (g_deterministic_mode && tc.split_k > 1)
            tc.split_k = 1;

        plan.tile_id = static_cast<int>(tc.tile);
        plan.split_k = tc.split_k;

        int bm = 64;
        int bn = 128;
        tileShape(tc.tile, bm, bn);

        if constexpr (CB == 0)
        {
            if (!g_deterministic_mode && tc.split_k == 1 && g_stream_k_force_mode == 2)
            {
                const int ntx = (N + bn - 1) / bn;
                const int nty = (M + bm - 1) / bm;
                const int total_tiles = ntx * nty;
                plan.streamk = 2;
                plan.streamk_fixup_bytes = static_cast<size_t>(total_tiles) * bm * bn * sizeof(float);
                return plan;
            }
            if (!g_deterministic_mode && tc.split_k == 1 && shouldUseStreamK(M, N, K, bm, bn, prefill_ctx))
            {
                plan.streamk = 1;
                return plan;
            }
        }

        if (tc.split_k > 1)
            plan.splitk_partials_bytes = static_cast<size_t>(tc.split_k) * M * N * sizeof(float);
        return plan;
    }

    // =========================================================================
    // Unified prefill dispatch: single format-agnostic path for ALL codebooks.
    //
    // Dispatch priority:
    //   1. BK256 path (CB=0 only): large-K shapes where BK64 can't fill GPU
    //   2. Force-tile override (g_force_tile_id): for sweep benchmarks
    //   3. Total heuristic dispatch (choosePrefillTile with format complexity)
    //   4. Tile launch: StreamK evaluation (CB=0 only) → standard split_k
    // =========================================================================
    template <uint8_t CB>
    bool launchGenericPrefillBK64(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const int32_t *d_sums_A_block,
        int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int serial_m1_k_partitions,
        int serial_m1_uses_ordered_reducer,
        cudaStream_t cuda_stream,
        CUDAPrefillContext_ *prefill_ctx)
    {
        Q40PrefillRoute q40_route = Q40PrefillRoute::Generic;

        // ─── BK256 path (CB=0 only) ──────────────────────────────────
        // BK256 processes 256 K-elements per outer tile (4× fewer K-iterations
        // than BK64), benefiting K-heavy shapes like FFN_Down (K=18944).
        // Uses 1 block/SM occupancy, so only helps when BK64 can't fill the GPU.
        if constexpr (CB == 0)
        {
            q40_route = chooseQ40PrefillRoute(M, N, K, prefill_ctx);
            if (q40_route == Q40PrefillRoute::BK256Narrow ||
                q40_route == Q40PrefillRoute::BK256Wide)
            {
                constexpr int sk = 1;
                if (q40_route == Q40PrefillRoute::BK256Narrow)
                {
                    recordLastLaunchSelection(-3, sk, true, 0);
                    return launchNativeVNNITC_BK256<128, 64, 4, 2, sk>(
                        d_A_int8, d_payload, d_scales, d_C_fp32,
                        d_scales_A_block, M, N, K, alpha, beta,
                        d_C_existing, d_bias, cuda_stream, prefill_ctx,
                        serial_m1_k_partitions,
                        serial_m1_uses_ordered_reducer);
                }

                recordLastLaunchSelection(-2, sk, true, 0);
                return launchNativeVNNITC_BK256<128, 128, 4, 4, sk>(
                    d_A_int8, d_payload, d_scales, d_C_fp32,
                    d_scales_A_block, M, N, K, alpha, beta,
                    d_C_existing, d_bias, cuda_stream, prefill_ctx,
                    serial_m1_k_partitions,
                    serial_m1_uses_ordered_reducer);
            }
        }

        // ─── Tile selection (3-tier priority) ─────────────────────────
        TileChoice tc;

        if constexpr (CB == 0)
        {
            if (q40_route == Q40PrefillRoute::ProfiledT64x64)
            {
                tc = {TileId::T64x64_w2x2, 1};
            }
            else if (g_force_tile_id >= 0 && g_force_tile_id <= 5)
            {
                tc = {static_cast<TileId>(g_force_tile_id),
                      (g_force_split_k > 0) ? g_force_split_k : 1};
            }
            else
            {
                constexpr FormatComplexity complexity =
                    getFormatComplexity(CB);
                tc = choosePrefillTile(
                    M, N, K, prefill_ctx, complexity, CB);
            }
        }
        else if (g_force_tile_id >= 0 && g_force_tile_id <= 5)
        {
            // Force-tile: bypass everything for sweep benchmarks
            tc = {static_cast<TileId>(g_force_tile_id),
                  (g_force_split_k > 0) ? g_force_split_k : 1};
        }
        else
        {
            constexpr FormatComplexity complexity = getFormatComplexity(CB);
            tc = choosePrefillTile(
                M, N, K, prefill_ctx, complexity, CB);
        }

        // Deterministic mode also clamps an explicitly forced diagnostic. Auto
        // dispatch already returns split_k=1 for every production geometry.
        if (g_deterministic_mode && tc.split_k > 1)
            tc.split_k = 1;

        // ─── Tile launch with optional diagnostic Stream-K (CB=0) ─────
#define DISPATCH_TILE_SK(BM_, BN_, WM_, WN_)                                            \
    do                                                                                  \
    {                                                                                   \
        if constexpr (CB == 0)                                                          \
        {                                                                               \
            if (!g_deterministic_mode && tc.split_k == 1 && g_stream_k_force_mode == 2)   \
            {                                                                           \
                recordLastLaunchSelection(static_cast<int>(tc.tile), 1, false, 2);      \
                return launchNativeVNNITC_BK64_StreamK_TwoPass<CB, BM_, BN_, WM_, WN_>( \
                    d_A_int8, d_payload, d_scales, d_mins, d_emins,                     \
                    d_C_fp32, d_scales_A_block, d_sums_A_block,                         \
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream,            \
                    prefill_ctx);                                                       \
            }                                                                           \
            if (!g_deterministic_mode && tc.split_k == 1 && shouldUseStreamK(M, N, K, BM_, BN_, prefill_ctx)) \
            {                                                                           \
                recordLastLaunchSelection(static_cast<int>(tc.tile), 1, false, 1);      \
                return launchNativeVNNITC_BK64_StreamK<CB, BM_, BN_, WM_, WN_>(         \
                    d_A_int8, d_payload, d_scales, d_mins, d_emins,                     \
                    d_C_fp32, d_scales_A_block, d_sums_A_block,                         \
                    M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream,            \
                    prefill_ctx);                                                       \
            }                                                                           \
        }                                                                               \
        switch (tc.split_k)                                                             \
        {                                                                               \
        case 8:                                                                         \
            recordLastLaunchSelection(static_cast<int>(tc.tile), 8, false, 0);          \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 8>(                  \
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32,               \
                d_scales_A_block, d_sums_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, \
                cuda_stream, prefill_ctx, serial_m1_k_partitions,                       \
                serial_m1_uses_ordered_reducer);                                        \
        case 4:                                                                         \
            recordLastLaunchSelection(static_cast<int>(tc.tile), 4, false, 0);          \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 4>(                  \
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32,               \
                d_scales_A_block, d_sums_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, \
                cuda_stream, prefill_ctx, serial_m1_k_partitions,                       \
                serial_m1_uses_ordered_reducer);                                        \
        case 2:                                                                         \
            recordLastLaunchSelection(static_cast<int>(tc.tile), 2, false, 0);          \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 2>(                  \
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32,               \
                d_scales_A_block, d_sums_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, \
                cuda_stream, prefill_ctx, serial_m1_k_partitions,                       \
                serial_m1_uses_ordered_reducer);                                        \
        default:                                                                        \
            recordLastLaunchSelection(static_cast<int>(tc.tile), 1, false, 0);          \
            return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, 1>(                  \
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32,               \
                d_scales_A_block, d_sums_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, \
                cuda_stream, prefill_ctx, serial_m1_k_partitions,                       \
                serial_m1_uses_ordered_reducer);                                        \
        }                                                                               \
    } while (0)

        switch (tc.tile)
        {
        case TileId::T64x64_w2x2:
            DISPATCH_TILE_SK(64, 64, 2, 2);
        case TileId::T64x128_w2x2:
            DISPATCH_TILE_SK(64, 128, 2, 2);
        case TileId::T64x128_w4x2:
            DISPATCH_TILE_SK(64, 128, 4, 2);
        case TileId::T64x128_w2x4:
            DISPATCH_TILE_SK(64, 128, 2, 4);
        case TileId::T128x128_w4x2:
            DISPATCH_TILE_SK(128, 128, 4, 2);
        case TileId::T128x128_w4x4:
            DISPATCH_TILE_SK(128, 128, 4, 4);
        }

#undef DISPATCH_TILE_SK
        return false; // unreachable
    }
}

extern "C"
{
    // Stream-K force mode: 0=auto(heuristic), 1=force ON, -1=force OFF
    void cudaNativeVNNIPrefill_setStreamKMode(int mode)
    {
        g_stream_k_force_mode = mode;
    }

    int cudaNativeVNNIPrefill_getStreamKMode()
    {
        return g_stream_k_force_mode;
    }

    // BK256 force mode: 0=auto(heuristic), 1=force ON, -1=force OFF
    void cudaNativeVNNIPrefill_setBK256Mode(int mode)
    {
        g_bk256_force_mode = mode;
    }

    int cudaNativeVNNIPrefill_getBK256Mode()
    {
        return g_bk256_force_mode;
    }

    // Deterministic mode API: disables stream-K and enables parity-preserving
    // dispatch rules in the BK64 prefill path.
    void cudaNativeVNNIPrefill_setDeterministicMode(bool enabled)
    {
        g_deterministic_mode = enabled;
    }

    bool cudaNativeVNNIPrefill_getDeterministicMode()
    {
        return g_deterministic_mode;
    }

    void cudaNativeVNNIPrefill_getLastLaunchSelection(
        int *tile_id,
        int *split_k,
        int *used_bk256,
        int *used_streamk)
    {
        if (tile_id)
            *tile_id = g_last_launch_selection.tile_id;
        if (split_k)
            *split_k = g_last_launch_selection.split_k;
        if (used_bk256)
            *used_bk256 = g_last_launch_selection.used_bk256;
        if (used_streamk)
            *used_streamk = g_last_launch_selection.used_streamk;
    }

    // Force-tile/split-k override for sweep benchmarks.
    //   tile_id: -1 = auto, 0..5 = TileId enum (T64x64_w2x2 .. T128x128_w4x4)
    //   split_k: 0 = auto, 1..8 = forced split-K value
    void cudaNativeVNNIPrefill_setForceTile(int tile_id, int split_k)
    {
        g_force_tile_id = tile_id;
        g_force_split_k = split_k;
    }

    void cudaNativeVNNIPrefill_getForceTile(int *tile_id, int *split_k)
    {
        if (tile_id)
            *tile_id = g_force_tile_id;
        if (split_k)
            *split_k = g_force_split_k;
    }

    // Return the number of tiles for a given tile config + shape
    int cudaNativeVNNIPrefill_getTileCount(int tile_id, int M, int N)
    {
        int bm = 64, bn = 64;
        switch (static_cast<TileId>(tile_id))
        {
        case TileId::T64x64_w2x2:
            bm = 64;
            bn = 64;
            break;
        case TileId::T64x128_w2x2:
        case TileId::T64x128_w4x2:
        case TileId::T64x128_w2x4:
            bm = 64;
            bn = 128;
            break;
        case TileId::T128x128_w4x2:
        case TileId::T128x128_w4x4:
            bm = 128;
            bn = 128;
            break;
        }
        return ((M + bm - 1) / bm) * ((N + bn - 1) / bn);
    }
    // -----------------------------------------------------------------
    // Prefill context lifetime
    // -----------------------------------------------------------------
    CUDAPrefillContext *cudaPrefillContext_create(int device_id)
    {
        auto *ctx = new (std::nothrow) CUDAPrefillContext_();
        if (!ctx)
            return nullptr;
        ctx->device_id = device_id;
        return ctx;
    }

    void cudaPrefillContext_destroy(CUDAPrefillContext *ctx)
    {
        if (!ctx)
            return;
        delete ctx;
    }

    void cudaPrefillContext_bindWorkspace(
        CUDAPrefillContext *ctx,
        float *splitk_partials,
        size_t splitk_partials_bytes,
        float *streamk_fixup,
        size_t streamk_fixup_bytes)
    {
        if (!ctx)
            return;
        ctx->workspace_splitk_partials = splitk_partials;
        ctx->workspace_splitk_partials_size = splitk_partials_bytes;
        ctx->workspace_fixup_buf = streamk_fixup;
        ctx->workspace_fixup_buf_size = streamk_fixup_bytes;
    }

    bool cudaNativeVNNIPrefill_getWorkspacePlan(
        uint8_t codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *splitk_partials_bytes,
        size_t *streamk_fixup_bytes,
        int *planned_split_k,
        int *planned_streamk)
    {
        if (splitk_partials_bytes)
            *splitk_partials_bytes = 0;
        if (streamk_fixup_bytes)
            *streamk_fixup_bytes = 0;
        if (planned_split_k)
            *planned_split_k = 1;
        if (planned_streamk)
            *planned_streamk = 0;
        if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;

        CUDAPrefillContext_ temp_ctx;
        temp_ctx.device_id = cuda_device_id;

        PrefillWorkspacePlan plan;
        switch (codebook_id)
        {
        case 0:
            plan = planGenericPrefillWorkspace<0>(M, N, K, &temp_ctx);
            break;
        case 4:
            plan = planGenericPrefillWorkspace<4>(M, N, K, &temp_ctx);
            break;
        case 5:
            plan = planGenericPrefillWorkspace<5>(M, N, K, &temp_ctx);
            break;
        case 6:
            plan = planGenericPrefillWorkspace<6>(M, N, K, &temp_ctx);
            break;
        case 7:
            plan = planGenericPrefillWorkspace<7>(M, N, K, &temp_ctx);
            break;
        case 8:
            plan = planGenericPrefillWorkspace<8>(M, N, K, &temp_ctx);
            break;
        case 9:
            plan = planGenericPrefillWorkspace<9>(M, N, K, &temp_ctx);
            break;
        case 10:
            plan = planGenericPrefillWorkspace<10>(M, N, K, &temp_ctx);
            break;
        case 11:
            plan = planGenericPrefillWorkspace<11>(M, N, K, &temp_ctx);
            break;
        case 12:
            plan = planGenericPrefillWorkspace<12>(M, N, K, &temp_ctx);
            break;
        case 13:
            plan = planGenericPrefillWorkspace<13>(M, N, K, &temp_ctx);
            break;
        case 14:
            plan = planGenericPrefillWorkspace<14>(M, N, K, &temp_ctx);
            break;
        case 15:
            plan = planGenericPrefillWorkspace<15>(M, N, K, &temp_ctx);
            break;
        case 16:
            plan = planGenericPrefillWorkspace<16>(M, N, K, &temp_ctx);
            break;
        case 17:
            plan = planGenericPrefillWorkspace<17>(M, N, K, &temp_ctx);
            break;
        case 19:
            plan = planGenericPrefillWorkspace<19>(M, N, K, &temp_ctx);
            break;
        default:
            return false;
        }

        if (splitk_partials_bytes)
            *splitk_partials_bytes = plan.splitk_partials_bytes;
        if (streamk_fixup_bytes)
            *streamk_fixup_bytes = plan.streamk_fixup_bytes;
        if (planned_split_k)
            *planned_split_k = plan.split_k;
        if (planned_streamk)
            *planned_streamk = plan.streamk;
        return true;
    }
} // extern "C"

// Profitability gate removed: NativeVNNI is now the only CUDA GEMM path.
// TC/CUTLASS expanded fallback has been sunset. All codebooks always use
// the NativeVNNI prefill kernel regardless of shape.

extern "C" bool cudaNativeVNNIPrefill_fp32(
    const int8_t *d_A_int8,
    const uint8_t *d_payload,
    const uint16_t *d_scales,
    const uint16_t *d_mins,
    const uint32_t *d_emins,
    float *d_C_fp32,
    const float *d_scales_A_block,
    const int32_t *d_sums_A_block,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    const float *d_C_existing,
    const float *d_bias,
    uint8_t codebook_id,
    int cuda_device_id,
    void *stream,
    CUDAPrefillContext *prefill_ctx)
{
    if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
        return false;
    if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
        return false;
    if (!prefill_ctx)
        return false;
    if (!isAmperePlus(cuda_device_id))
        return false;
    if (cudaSetDevice(cuda_device_id) != cudaSuccess)
        return false;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int serial_m1_uses_ordered_reducer = 0;
    int serial_m1_k_partitions = 0;
    if (!cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
            codebook_id,
            N,
            K,
            querySmCount(prefill_ctx),
            &serial_m1_uses_ordered_reducer,
            &serial_m1_k_partitions))
    {
        return false;
    }

    bool ok = false;
    switch (codebook_id)
    {
    // --- Single-scale formats (no min correction) ---
    case 0:
        ok = launchGenericPrefillBK64<0>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 4:
        ok = launchGenericPrefillBK64<4>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 6: // Q5_0
        ok = launchGenericPrefillBK64<6>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 11: // IQ3_S
        ok = launchGenericPrefillBK64<11>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 12: // IQ3_XXS
        ok = launchGenericPrefillBK64<12>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 15: // IQ2_XXS
        ok = launchGenericPrefillBK64<15>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;

    // --- Asymmetric formats (need min correction, d_mins required) ---
    case 5: // Q4_1 / Q4_K
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<5>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            d_sums_A_block, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 7: // Q5_1 / Q5_K
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<7>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            d_sums_A_block, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 16: // IQ1_S
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<16>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            d_sums_A_block, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;

    // --- Dual-scale formats (separate lo/hi scales via split MMA) ---
    case 8: // Q6_K
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<8>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 9: // Q3_K
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<9>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 10: // Q2_K (dual-scale + asymmetric via emins)
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<10>(
            d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 13: // IQ2_S
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<13>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 14: // IQ2_XS
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<14>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;
    case 17: // IQ1_M (dual-scale + delta correction)
        if (!d_mins)
            return false;
        ok = launchGenericPrefillBK64<17>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;

    // --- 8-bit format (no decode overhead, single-scale) ---
    case 19: // Q8_0
        ok = launchGenericPrefillBK64<19>(
            d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block,
            nullptr, M, N, K, alpha, beta, d_C_existing, d_bias,
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer,
            cuda_stream, prefill_ctx);
        break;

    default:
        return false;
    }

    if (ok && llaminar2::PerfStatsCollector::isEnabled())
    {
        int tile_id = -1;
        int split_k = 1;
        int used_bk256 = 0;
        int used_streamk = 0;
        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &tile_id, &split_k, &used_bk256, &used_streamk);
        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "cuda_native_vnni_prefill_calls",
            1.0,
            "gemm",
            "cuda:" + std::to_string(cuda_device_id),
            llaminar2::PerfStatsCollector::Tags{
                {"codebook", std::to_string(static_cast<int>(codebook_id))},
                {"m", std::to_string(M)},
                {"n", std::to_string(N)},
                {"k", std::to_string(K)},
                {"tile_id", std::to_string(tile_id)},
                {"split_k", std::to_string(split_k)},
                {"bk256", used_bk256 ? "1" : "0"},
                {"streamk", std::to_string(used_streamk)},
                {"sums_a", d_sums_A_block ? "1" : "0"}});
    }
    return ok;
}
