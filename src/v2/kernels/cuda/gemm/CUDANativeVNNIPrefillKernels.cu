/**
 * @file CUDANativeVNNIPrefillKernels.cu
 * @brief All-format native-VNNI tensor-core prefill kernels for CUDA.
 *
 * Weights stay in their compact native payload form in VRAM. Each CTA decodes
 * one format-specific payload tile into transient shared-memory INT8 values,
 * then feeds the common `mma.sync.m16n8k32` fragment path. Launch policy is
 * responsible for exposing enough independent output tiles to fill the GPU
 * while retaining the public M=1 FP32 reduction tree for every output value.
 * The only K-parallel schedule is the public M=1 partition schedule followed
 * by the same ascending ordered reducer. Independent partial trees and atomic
 * accumulation are absent because neither can satisfy byte equivalence.
 * Optional activation sums are block-major (K/32 by physical M), published
 * directly by both CUDA quantizers. Scales and activation bytes remain
 * row-major; no extra transpose, workspace, or metadata copy is required.
 * The BK64 arithmetic and staged-operand candidates live in the shared
 * CUDANativeVNNIPrefillDevice.cuh body; this file owns public planning, launch
 * selection and resource queries. Production staging remains RegisterDecode
 * until an authenticated candidate tournament authorizes a different schedule.
 * CUDA function attributes belong to the current device context, not the host
 * process. Launch preparation applies them idempotently for that context; no
 * process-global readiness bit may outlive or alias a device/module lifetime.
 */

#include <cuda_runtime.h>

#include "CUDADenseProductionPrefillOverlayGenerated.inc"
#include "CUDANativeVNNIPrefillDiagnostics.h"
#include "CUDANativeVNNIPrefillDevice.cuh"
#include "CUDANativeVNNIDecodeCommon.cuh"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
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

/** @brief Return whether the generated serial policy owns this codebook. */
extern "C" bool cudaNativeVNNIGemvTuned_supportsCodebook(
    uint8_t codebook_id);

// =========================================================================
// Per-device prefill context. Owned by KernelFactory, one per device.
// =========================================================================
struct CUDAPrefillContext_
{
    int sm_count = 0;
    int device_id = -1;
    float *workspace_canonical_kpart_partials = nullptr;
    size_t workspace_canonical_kpart_partials_size = 0;
};

struct LastLaunchSelection_
{
    int tile_id = -1;
    int k_partitions = 1;
    int used_bk256 = 0;
    int used_canonical_kpart = 0;
    int used_exact_overlay = 0;
    llaminar2::cuda::prefill::PrefillStagingSchedule staging =
        llaminar2::cuda::prefill::PrefillStagingSchedule::RegisterDecode;
};

// Thread-local because tile sweep benchmarks can launch from multiple worker
// threads. Diagnostics read back the selection on the same thread.
static thread_local LastLaunchSelection_ g_last_launch_selection;

static inline void recordLastLaunchSelection(
    int tile_id,
    int k_partitions,
    bool used_bk256,
    bool used_canonical_kpart = false,
    bool used_exact_overlay = false,
    llaminar2::cuda::prefill::PrefillStagingSchedule staging =
        llaminar2::cuda::prefill::PrefillStagingSchedule::RegisterDecode)
{
    g_last_launch_selection.tile_id = tile_id;
    g_last_launch_selection.k_partitions = k_partitions;
    g_last_launch_selection.used_bk256 = used_bk256 ? 1 : 0;
    g_last_launch_selection.used_canonical_kpart =
        used_canonical_kpart ? 1 : 0;
    g_last_launch_selection.used_exact_overlay =
        used_exact_overlay ? 1 : 0;
    g_last_launch_selection.staging = staging;
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

static float *getCanonicalKpartPartials(
    CUDAPrefillContext_ *ctx,
    size_t required_bytes,
    cudaStream_t stream)
{
    // The ordered reducer consumes every public-M1 partition slot. Some legal
    // schedules have trailing empty partitions, so stale workspace contents
    // must not survive from an earlier request or projection.
    if (ctx->workspace_canonical_kpart_partials &&
        ctx->workspace_canonical_kpart_partials_size >= required_bytes)
    {
        cudaMemsetAsync(
            ctx->workspace_canonical_kpart_partials,
            0,
            required_bytes,
            stream);
        return ctx->workspace_canonical_kpart_partials;
    }
    return nullptr;
}

namespace
{
    using llaminar2::cuda_native_vnni::fp16_bits_to_float;

    using namespace llaminar2::cuda::prefill;

    // ─── Sweep-derived tile dispatch ───────────────────────────────────
    // Tile configurations validated across production geometries. K reduction
    // is not a free tuning dimension: only full-K and canonical public-M1
    // partition schedules are admissible.

    enum class TileId : uint8_t
    {
        T64x64_w2x2,   // BM=64  BN=64  WM=2 WN=2  (128 threads)
        T64x128_w2x2,  // BM=64  BN=128 WM=2 WN=2  (128 threads)
        T64x128_w4x2,  // BM=64  BN=128 WM=4 WN=2  (256 threads)
        T64x128_w2x4,  // BM=64  BN=128 WM=2 WN=4  (256 threads)
        T128x128_w4x2, // BM=128 BN=128 WM=4 WN=2  (256 threads)
        T128x128_w4x4, // BM=128 BN=128 WM=4 WN=4  (512 threads)
        Count,
    };

    constexpr int kDensePrefillTileCount = static_cast<int>(TileId::Count);

    /** Return whether an integer names one registered BK64 tile geometry. */
    constexpr bool isDensePrefillTileId(int tile_id)
    {
        return tile_id >= 0 && tile_id < kDensePrefillTileCount;
    }

    struct TileChoice
    {
        TileId tile;
    };

    // Fold public-M1 K-partition partials into the final output. The loop order
    // is part of the byte-equivalence contract and must remain ascending.
    __global__ void canonical_kpart_reduce(
        const float *__restrict__ partials,
        float *__restrict__ C,
        const float *__restrict__ C_existing,
        const float *__restrict__ bias,
        int M, int N, int k_partitions,
        float beta)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = M * N;
        if (idx >= total)
            return;

        float sum = 0.0f;
        for (int z = 0; z < k_partitions; ++z)
            sum += partials[z * total + idx];

        if (beta != 0.0f && C_existing)
            sum += beta * C_existing[idx];
        if (bias)
            sum += bias[idx % N];

        C[idx] = sum;
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

    template <int BM, int BN, int WARPS_M, int WARPS_N,
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
        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N;
        const int wc = warp_id % WARPS_N;
        const int gid = lane_id >> 2;

        const int block_m = blockIdx.x * BM;
        const int block_n = blockIdx.y * BN;

        const int num_q40_blocks = K / 32;
        const int num_outer_tiles = (num_q40_blocks + 7) / 8;
        constexpr int ot_begin = 0;
        const int ot_end = num_outer_tiles;

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

                    C[out_idx0] =
                        __fmul_rn(acc[wi][wj][0], output_alpha);
                    C[out_idx1] =
                        __fmul_rn(acc[wi][wj][1], output_alpha);
                    C[out_idx2] =
                        __fmul_rn(acc[wi][wj][2], output_alpha);
                    C[out_idx3] =
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

                        if (beta != 0.0f && C_existing)
                            val += beta * C_existing[out_idx];
                        if (bias)
                            val += (e & 1) ? bias1 : bias0;
                        C[out_idx] = val;
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
    // Tile force: override output geometry for sweep benchmarks. K reduction
    // policy is intentionally not forceable independently of public M=1.
    // =========================================================================
    static int g_force_tile_id = []()
    {
        return llaminar2::debugEnv().gemm.cuda_force_prefill_tile;
    }();
    // Candidate control for the economical K-parallel implementation. This
    // route inherits the exact public M=1 partition count and boundaries
    // instead of accepting an independent reduction geometry.
    static bool g_force_canonical_kpart = false;

    // A diagnostic scope may change operand delivery on its own host thread.
    // Normal inference retains the certified default; no implicit promotion is
    // made from a successful resource query or an isolated timing sample.
    static thread_local PrefillStagingSchedule g_staging_schedule =
        PrefillStagingSchedule::RegisterDecode;

    // Profilers must measure the generic policy and every concrete candidate
    // independently of any previously installed exact cells. Production keeps
    // this enabled for the lifetime of the process; only the isolated sweep
    // harness scopes it off while gathering replacement evidence.
    static bool g_dense_prefill_exact_overlay_enabled = true;

    // BK256 mode: 0=auto (heuristic), 1=force ON, -1=force OFF
    // Set via LLAMINAR_BK256_MODE env var or extern C API.
    static int g_bk256_force_mode = []()
    {
        return llaminar2::debugEnv().gemm.cuda_bk256_mode;
    }();

    /**
     * @brief Result of consulting the installed dense-prefill exact overlay.
     *
     * `Invalid` is deliberately distinct from `NotSelected`: a malformed
     * generated entry is a production-policy failure and must never silently
     * continue through the generic heuristic. `NotSelected` means either that
     * the runtime key was not explicitly swept or that a profiler force control
     * owns this process-local launch.
     */
    enum class DensePrefillOverlayStatus : uint8_t
    {
        NotSelected,
        Selected,
        Invalid,
    };

    /**
     * @brief Return whether an identifier has a compiled NativeVNNI prefill path.
     *
     * Workspace planning and launch dispatch must accept exactly the same
     * codebook set. Centralizing the inventory prevents a newly implemented
     * format from becoming launchable without also receiving a valid persistent
     * workspace contract.
     */
    constexpr bool isSupportedNativeVNNIPrefillCodebook(uint8_t codebook)
    {
        switch (codebook)
        {
        case 0:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 19:
        case llaminar2::kNativeVnniExpandedInt8MinCodebook:
            return true;
        default:
            return false;
        }
    }

    /**
     * @brief Validate one generated overlay tuple before it controls execution.
     *
     * BK256 launch identifiers exist only for Q4_0 and cannot preserve the
     * canonical public-M1 partition tree. Ordinary BK64 tiles cover every
     * supported codebook and may select either direct or canonical arithmetic.
     */
    constexpr bool isValidDensePrefillOverlayConfig(
        uint8_t codebook,
        const llaminar2::cuda::generated::CUDADensePrefillOverlayConfig &config)
    {
        if (!isKnownPrefillStagingSchedule(config.staging))
            return false;
        const bool ordinary_tile = isDensePrefillTileId(config.tile_id);
        const bool valid_bk256 =
            codebook == 0 && config.bk256 && !config.canonical_kpart &&
            config.staging == PrefillStagingSchedule::RegisterDecode &&
            (config.tile_id == -3 || config.tile_id == -2);
        const bool valid_bk64 = !config.bk256 && ordinary_tile;
        return valid_bk256 || valid_bk64;
    }

    /**
     * @brief Resolve one authenticated dense-prefill launch before capture.
     *
     * Exact overlays are keyed only by the runtime-visible codebook and matrix
     * geometry `(M,N,K)`. Profiler force controls have higher precedence so the
     * sweep can continue to measure every physical candidate after an overlay
     * has been installed. Ordinary production has no such controls and sees the
     * exact overlay before the total generic policy.
     *
     * @param codebook Runtime NativeVNNI execution codebook.
     * @param M Number of prefill rows in the captured bucket.
     * @param N Projection output width.
     * @param K Projection reduction width.
     * @param config Receives the concrete launch tuple on `Selected`.
     * @return Selection status, retaining malformed generated state as fatal.
     */
    DensePrefillOverlayStatus selectDensePrefillOverlay(
        uint8_t codebook,
        int M,
        int N,
        int K,
        llaminar2::cuda::generated::CUDADensePrefillOverlayConfig &config)
    {
        const bool profiler_override =
            !g_dense_prefill_exact_overlay_enabled ||
            g_force_canonical_kpart ||
            g_staging_schedule != PrefillStagingSchedule::RegisterDecode ||
            g_force_tile_id >= 0 ||
            g_bk256_force_mode != 0;
        if (profiler_override)
            return DensePrefillOverlayStatus::NotSelected;

        if (!llaminar2::cuda::generated::selectCUDADensePrefillOverlay(
                codebook, M, N, K, config))
        {
            return DensePrefillOverlayStatus::NotSelected;
        }

        return isValidDensePrefillOverlayConfig(codebook, config)
                   ? DensePrefillOverlayStatus::Selected
                   : DensePrefillOverlayStatus::Invalid;
    }

    /**
     * @brief Query bytes for one row count using the public M=1 reduction tree.
     *
     * The partition boundaries are owned by the serial-decode policy. Prefill
     * may reuse that arithmetic only after this query proves an ordered reducer
     * with more than one partition on the selected device.
     */
    bool queryCanonicalPrefillWorkspaceForRows(
        uint8_t codebook_id,
        int rows,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions)
    {
        CUDAPrefillContext_ temp_ctx;
        temp_ctx.device_id = cuda_device_id;

        int uses_ordered_reducer = 0;
        int k_partitions = 0;
        if (!cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
                codebook_id,
                N,
                K,
                querySmCount(&temp_ctx),
                &uses_ordered_reducer,
                &k_partitions) ||
            uses_ordered_reducer == 0 || k_partitions <= 1)
        {
            return false;
        }

        const size_t partitions = static_cast<size_t>(k_partitions);
        const size_t row_count = static_cast<size_t>(rows);
        const size_t columns = static_cast<size_t>(N);
        constexpr size_t element_bytes = sizeof(float);
        if (row_count > std::numeric_limits<size_t>::max() / partitions ||
            row_count * partitions >
                std::numeric_limits<size_t>::max() / columns ||
            row_count * partitions * columns >
                std::numeric_limits<size_t>::max() / element_bytes)
        {
            return false;
        }

        if (canonical_kpart_partials_bytes)
        {
            *canonical_kpart_partials_bytes =
                partitions * row_count * columns * element_bytes;
        }
        if (planned_k_partitions)
            *planned_k_partitions = k_partitions;
        return true;
    }

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
         * Qwen3.6 35B MoE expert gate/up projection, verifier row domain.
         *
         * The ordinary 64x128 policy exposes only four CTAs for N=512, leaving
         * most of an 82-SM GA102 idle throughout the K=2048 reduction. A full
         * all-format tournament over every M=2..16 plus the M=31 depth sentinel
         * proved that 64x64 preserves every result byte for all asymmetric and
         * dual-scale codebooks while exposing eight independent output tiles.
         * Depending on payload complexity, the measured speedup was 1.69x to
         * 2.21x. This is purely an output-geometry change and does not
         * re-parenthesize the canonical K sum.
         *
         * Keep the upper M boundary explicit. Rows above the verifier witness
         * domain remain total through the general prefill heuristic; extending
         * this overlay requires a new byte-exact tournament at those work sizes.
         */
        if (M >= 2 && M <= 31 && N == 512 && K == 2048)
            return {TileId::T64x64_w2x2};

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
                return {TileId::T64x128_w2x4};
            if (codebook == 16)
                return {TileId::T64x128_w4x2};
        }

        // Well-filling: enough output tiles to saturate SMs; prefer w2x2.
        if (t64x128 >= SM)
            return {TileId::T64x128_w2x2};

        /*
         * Underfilled shapes still retain the public M=1 arithmetic tree per
         * output row. Independently rounded partition sums would change
         * parenthesization, so occupancy comes from output geometry instead.
         */
        return {TileId::T64x128_w2x2};
    }

    // ─── Sweep-derived output-tile heuristic ──────────────────────────
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
        if (isDensePrefillTileId(g_force_tile_id))
            return {static_cast<TileId>(g_force_tile_id)};

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
                    return {TileId::T128x128_w4x4};
                return {TileId::T128x128_w4x2};
            }
            // Demote to 64×128
            return {pick_warp(t64x128, ki)};
        }

        // ═══ TIER 2: 64×128 fills ═══
        if (fills_64x128)
        {
            // Prefer 64×64 for small K + high tile parallelism
            if (ki <= 7 && t64 >= (5 * SM / 2))
                return {TileId::T64x64_w2x2};
            if (ki <= 14 && t64x128 < (13 * SM / 10) && t64 >= 2 * SM)
                return {TileId::T64x64_w2x2};

            const TileId warp = pick_warp(t64x128, ki);

            // 128×128 override for very large K at moderate M.
            if (M >= 256 && ki >= 40 && t128 >= 32 && t64x128 <= (3 * SM / 2))
                return {TileId::T128x128_w4x2};

            // Prefer the larger tile when its output grid remains economical.
            if (M >= 128 && ki >= 40 && t128 >= 32 && t64x128 >= (3 * SM / 2))
                return {TileId::T128x128_w4x2};

            return {warp};
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
                    return {TileId::T128x128_w4x2};

                return {warp};
            }
            return {TileId::T64x64_w2x2};
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
        return {tile};
    }

    /**
     * @brief Query the selected device without a mutable cross-device cache.
     * @param device_id CUDA ordinal whose prefill capability is being admitted.
     * @return True only for a successfully queried Ampere-or-newer device.
     *
     * Parallel participant capture can inspect different ordinals at once.
     * Caching an ordinal and result in separate statics races their publication.
     * This setup-time runtime query adds no node or work to captured replay.
     */
    bool isAmperePlus(int device_id)
    {
        int major = 0;
        return cudaDeviceGetAttribute(
            &major, cudaDevAttrComputeCapabilityMajor, device_id) == cudaSuccess &&
            major >= 8;
    }

    /** Map a compile-time output geometry back to its stable policy ID. */
    template <int BM, int BN, int WM, int WN>
    constexpr int densePrefillTileId()
    {
        if constexpr (BM == 64 && BN == 64 && WM == 2 && WN == 2)
            return 0;
        if constexpr (BM == 64 && BN == 128 && WM == 2 && WN == 2)
            return 1;
        if constexpr (BM == 64 && BN == 128 && WM == 4 && WN == 2)
            return 2;
        if constexpr (BM == 64 && BN == 128 && WM == 2 && WN == 4)
            return 3;
        if constexpr (BM == 128 && BN == 128 && WM == 4 && WN == 2)
            return 4;
        if constexpr (BM == 128 && BN == 128 && WM == 4 && WN == 4)
            return 5;
        return -1;
    }

    /**
     * @brief Select launch bounds that cannot force compiler spilling.
     *
     * The ordinary hint asks 128-thread tiles for three resident CTAs and
     * wider blocks for two. Exhaustive `cudaFuncGetAttributes` evidence on the
     * complete codebook/tile/reduction matrix showed that this cap forces local
     * memory for the cases below. Those exact specializations retain the same
     * kernel body and launch geometry but request only one resident CTA, which
     * lets ptxas allocate the registers the decode actually needs. Every other
     * specialization keeps its stronger occupancy contract. The setup-time
     * all-codebook resource gate remains authoritative and fails closed if a
     * future compiler changes either side of this table.
     */
    template <uint8_t CB, int BM, int BN, int WM, int WN,
              bool CanonicalKpart>
    constexpr int densePrefillMinBlocksHint()
    {
        constexpr int tile = densePrefillTileId<BM, BN, WM, WN>();
        static_assert(tile >= 0, "unregistered dense prefill tile geometry");
        constexpr int threads = WM * WN * 32;
        constexpr int ordinary_hint = threads >= 256 ? 2 : 3;

        if constexpr (CB == 10)
        {
            if constexpr (CanonicalKpart)
                return (tile == 0 || tile == 2) ? ordinary_hint : 1;
            return 1;
        }
        if constexpr (CB == 17)
        {
            if constexpr (CanonicalKpart)
                return (tile == 2 || tile == 3) ? ordinary_hint : 1;
            return tile == 2 ? ordinary_hint : 1;
        }
        if constexpr (CanonicalKpart &&
                      (CB == 0 || CB == 4 || CB == 6 || CB == 11 ||
                       CB == 12 || CB == 15 || CB == 19))
        {
            return tile <= 3 ? ordinary_hint : 1;
        }
        return (tile == 0 || tile == 2 || tile == 3)
                   ? ordinary_hint
                   : 1;
    }

    /** Return whether the relaxed specialization is physically spill-free. */
    template <uint8_t CB>
    constexpr bool densePrefillTileIsResourceEligible(
        TileId tile,
        bool canonical_kpart,
        PrefillStagingSchedule staging)
    {
        const int tile_id = static_cast<int>(tile);
        if (!isKnownPrefillStagingSchedule(staging))
            return false;
        if (staging != PrefillStagingSchedule::RegisterDecode)
        {
            constexpr int payload_bytes =
                llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
            if ((payload_bytes != 16 && payload_bytes != 20) || tile_id == 1)
                return false;
            // The complete production-module inventory proves eight bytes of
            // local storage for these direct asymmetric metadata schedules.
            // A forced/certified identity is rejected, never silently retiled.
            if constexpr (CB == 5 || CB == 7)
                if (!canonical_kpart && tile_id == 2 &&
                    (staging == PrefillStagingSchedule::AsyncWeightOperands ||
                     staging == PrefillStagingSchedule::AsyncAllOperands))
                    return false;
        }
        if constexpr (CB == 10 || CB == 17)
            return tile_id != 1 && tile_id != 4 && tile_id != 5;
        if constexpr (CB == 8 || CB == 9 || CB == 13 || CB == 14)
        {
            if (!canonical_kpart && (tile_id == 1 || tile_id == 4))
                return false;
        }
        return true;
    }

    /** @brief Visit the exact structural staging family without substitution.
     * @tparam CB Native codebook; its traits own the actual packed byte width.
     * @tparam BN Output columns in the tile.
     * @tparam Threads Copy/decode owners in one CTA.
     * @param staging Requested physical schedule.
     * @param operation Callable templated on the admitted schedule, returning bool.
     * @return False without invoking the operation for unsupported identities.
     * Resource queries and launches share this dispatch, including compile-time
     * exclusion of geometries that cannot provide one owner per block half.
     */
    template <uint8_t CB, int BN, int Threads, class Operation>
    bool visitDensePrefillStaging(
        PrefillStagingSchedule staging, Operation &&operation)
    {
        if (staging == PrefillStagingSchedule::RegisterDecode)
            return operation.template operator()<PrefillStagingSchedule::RegisterDecode>();
        constexpr int payload_bytes =
            llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
        if constexpr (supportsPrefillStaging(
                          PrefillStagingSchedule::AsyncPayload,
                          payload_bytes, 2, BN, Threads))
        {
            switch (staging)
            {
            case PrefillStagingSchedule::AsyncPayload:
                return operation.template operator()<PrefillStagingSchedule::AsyncPayload>();
            case PrefillStagingSchedule::AsyncWeightOperands:
                return operation.template operator()<PrefillStagingSchedule::AsyncWeightOperands>();
            case PrefillStagingSchedule::AsyncAllOperands:
                return operation.template operator()<PrefillStagingSchedule::AsyncAllOperands>();
            default:
                break;
            }
        }
        return false;
    }

    /** @brief Launch one exact staging specialization with CTA-local ordered K.
     * Buffers and stream are persistent caller-owned bindings; staging changes
     * neither them nor the public-M1 partition geometry and FP32 epilogue.
     */
    template <uint8_t CODEBOOK_ID, int BM, int BN, int WM, int WN,
              PrefillStagingSchedule Staging>
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
        int serial_m1_k_partitions = 1,
        int serial_m1_uses_ordered_reducer = 0)
    {
        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
        const dim3 block(WM * WN * 32);

        // Clear any stale CUDA error from prior operations (e.g. CUTLASS reference path)
        (void)cudaGetLastError();

        constexpr int block_size = WM * WN * 32;
        constexpr int min_blocks = densePrefillMinBlocksHint<
            CODEBOOK_ID, BM, BN, WM, WN, /*CanonicalKpart=*/false>();
        nativeVnniTC_BK64<
            CODEBOOK_ID,
            BM,
            BN,
            WM,
            WN,
            /*STAGES_=*/2,
            /*CANONICAL_KPART=*/false,
            block_size,
            min_blocks,
            Staging><<<grid, block, 0, cuda_stream>>>(
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
            M,
            N,
            K,
            alpha,
            beta,
            CanonicalM1PartitionGeometry{
                .count = serial_m1_k_partitions,
                .blocks_per_partition = (K / 32 + serial_m1_k_partitions - 1) /
                    serial_m1_k_partitions},
            serial_m1_uses_ordered_reducer);
        return cudaGetLastError() == cudaSuccess;
    }

    /**
     * @brief Launch tensor-core prefill with the public M=1 K-partition tree.
     *
     * Each grid-Z CTA owns exactly one partition from the generated serial
     * decode schedule. Partitions write disjoint FP32 partials, including an
     * explicit zero for an empty trailing partition, and
     * `canonical_kpart_reduce`
     * publishes them in ascending partition order. This recovers K-parallel
     * occupancy without introducing atomics or changing a single FP32
     * parenthesis relative to public M=1 decode.
     */
    template <uint8_t CODEBOOK_ID, int BM, int BN, int WM, int WN,
              PrefillStagingSchedule Staging>
    bool launchNativeVNNITC_BK64CanonicalKpart(
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
        CUDAPrefillContext_ *prefill_ctx,
        int serial_m1_k_partitions)
    {
        if (!prefill_ctx || serial_m1_k_partitions <= 1)
            return false;

        const size_t partials_bytes =
            static_cast<size_t>(serial_m1_k_partitions) * M * N *
            sizeof(float);
        float *partials = getCanonicalKpartPartials(
            prefill_ctx, partials_bytes, cuda_stream);
        if (!partials)
            return false;

        const dim3 grid(
            (M + BM - 1) / BM,
            (N + BN - 1) / BN,
            serial_m1_k_partitions);
        const dim3 block(WM * WN * 32);

        (void)cudaGetLastError();
        constexpr int block_size = WM * WN * 32;
        constexpr int min_blocks = densePrefillMinBlocksHint<
            CODEBOOK_ID, BM, BN, WM, WN, /*CanonicalKpart=*/true>();
        nativeVnniTC_BK64<
            CODEBOOK_ID,
            BM,
            BN,
            WM,
            WN,
            /*STAGES_=*/2,
            /*CANONICAL_KPART=*/true,
            block_size,
            min_blocks,
            Staging><<<grid, block, 0, cuda_stream>>>(
            d_A_int8,
            d_payload,
            d_scales,
            d_mins,
            d_emins,
            partials,
            d_scales_A_block,
            d_sums_A_block,
            d_C_existing,
            d_bias,
            M,
            N,
            K,
            alpha,
            beta,
            CanonicalM1PartitionGeometry{
                .count = serial_m1_k_partitions,
                .blocks_per_partition = (K / 32 + serial_m1_k_partitions - 1) /
                    serial_m1_k_partitions},
            /*serial_m1_uses_ordered_reducer=*/1);
        if (cudaGetLastError() != cudaSuccess)
            return false;

        const int total = M * N;
        constexpr int threads = 256;
        const int blocks = (total + threads - 1) / threads;
        canonical_kpart_reduce<<<blocks, threads, 0, cuda_stream>>>(
            partials,
            d_C_fp32,
            d_C_existing,
            d_bias,
            M,
            N,
            serial_m1_k_partitions,
            beta);
        return cudaGetLastError() == cudaSuccess;
    }

    /** Return the exact dynamic shared-memory footprint of one BK256 CTA. */
    template <int BM, int BN>
    constexpr int nativeVnniBK256SharedMemoryBytes()
    {
        constexpr int scales_b_offset =
            BM * BK128_STRIDE + BN * BK256_STRIDE;
        constexpr int scales_a_offset =
            scales_b_offset + 8 * BN * static_cast<int>(sizeof(uint16_t));
        constexpr int scales_a_aligned = (scales_a_offset + 3) & ~3;
        return scales_a_aligned + BM * 8 * static_cast<int>(sizeof(float));
    }

    /**
     * @brief Prepare the current context and capture one exact BK256 kernel.
     * @tparam BM Output rows per CTA.
     * @tparam BN Output columns per CTA.
     * @tparam WM Row warp count.
     * @tparam WN Column warp count.
     * @return False if context-local resource admission or launch fails.
     *
     * The wide specialization exceeds CUDA's default dynamic shared-memory
     * limit. Opt-in is function AND context local, including after context
     * reclamation. Apply it idempotently rather than maintaining another live
     * context ledger. This host preparation runs while constructing the graph;
     * graph replay executes the unchanged kernel with no additional nodes.
     */
    template <int BM, int BN, int WM, int WN>
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
        int serial_m1_k_partitions,
        int serial_m1_uses_ordered_reducer)
    {
        constexpr int smem_bytes =
            nativeVnniBK256SharedMemoryBytes<BM, BN>();

        /*
         * Ordinary and ordered-M1 kernels have intentionally different
         * register schedules, so configure and launch the exact specialization
         * selected by the immutable public-M1 arithmetic contract.
         */
        const auto kernel = serial_m1_uses_ordered_reducer
            ? nativeVnniTC_BK256<BM, BN, WM, WN, true>
            : nativeVnniTC_BK256<BM, BN, WM, WN, false>;
        if (cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_bytes) != cudaSuccess)
            return false;

        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);
        const dim3 block(WM * WN * 32);

        (void)cudaGetLastError(); // clear stale errors

        if (serial_m1_uses_ordered_reducer)
        {
            nativeVnniTC_BK256<
                BM,
                BN,
                WM,
                WN,
                true><<<grid, block, smem_bytes, cuda_stream>>>(
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
                false><<<grid, block, smem_bytes, cuda_stream>>>(
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
                beta,
                serial_m1_k_partitions);
        }
        return cudaGetLastError() == cudaSuccess;
    }

    using DensePrefillKernelResources = CUDADensePrefillKernelResources;

    /**
     * @brief Query setup-time resources for one exact kernel function.
     *
     * This function is intentionally outside the inference hot path. The
     * production tournament calls it after one untimed route-resolution probe
     * and before recording any timing event. Consequently a specialization
     * that allocates compiler local memory can fail closed without ever
     * becoming benchmark evidence or an installable dispatch target.
     */
    template <typename Kernel>
    bool queryDensePrefillKernelResources(
        Kernel kernel,
        int threads_per_block,
        size_t dynamic_shared_memory_bytes,
        DensePrefillKernelResources &resources)
    {
        cudaFuncAttributes attributes{};
        if (cudaFuncGetAttributes(&attributes, kernel) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        int active_blocks = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks,
                kernel,
                threads_per_block,
                dynamic_shared_memory_bytes) != cudaSuccess)
        {
            (void)cudaGetLastError();
            return false;
        }

        resources.kernel_symbol = reinterpret_cast<const void *>(kernel);
        resources.registers_per_thread = attributes.numRegs;
        resources.local_memory_bytes_per_thread = attributes.localSizeBytes;
        resources.static_shared_memory_bytes = attributes.sharedSizeBytes;
        resources.dynamic_shared_memory_bytes = dynamic_shared_memory_bytes;
        resources.threads_per_block = threads_per_block;
        resources.max_threads_per_block = attributes.maxThreadsPerBlock;
        resources.max_active_blocks_per_sm = active_blocks;
        return active_blocks > 0;
    }

    /** Query one BK64 primary and its optional ordered-reducer kernel. */
    template <uint8_t CB, int BM, int BN, int WM, int WN,
              PrefillStagingSchedule Staging>
    bool queryDensePrefillBK64Resources(
        bool canonical_kpart,
        DensePrefillKernelResources &primary,
        DensePrefillKernelResources &auxiliary)
    {
        constexpr int threads = WM * WN * 32;
        bool primary_ok = false;
        if (canonical_kpart)
        {
            constexpr int min_blocks = densePrefillMinBlocksHint<
                CB, BM, BN, WM, WN, /*CanonicalKpart=*/true>();
            primary_ok = queryDensePrefillKernelResources(
                nativeVnniTC_BK64<
                    CB,
                    BM,
                    BN,
                    WM,
                    WN,
                    /*STAGES_=*/2,
                    /*CANONICAL_KPART=*/true,
                    threads,
                    min_blocks,
                    Staging>,
                threads,
                0,
                primary);
            if (!primary_ok)
                return false;
            return queryDensePrefillKernelResources(
                canonical_kpart_reduce,
                /*threads_per_block=*/256,
                0,
                auxiliary);
        }

        constexpr int min_blocks = densePrefillMinBlocksHint<
            CB, BM, BN, WM, WN, /*CanonicalKpart=*/false>();
        primary_ok = queryDensePrefillKernelResources(
            nativeVnniTC_BK64<
                CB,
                BM,
                BN,
                WM,
                WN,
                /*STAGES_=*/2,
                /*CANONICAL_KPART=*/false,
                threads,
                min_blocks,
                Staging>,
            threads,
            0,
            primary);
        auxiliary = {};
        return primary_ok;
    }

    /** Query one registered, forceable BK64 output geometry. */
    template <uint8_t CB>
    bool queryDensePrefillTileResources(
        int tile_id,
        bool canonical_kpart,
        PrefillStagingSchedule staging,
        DensePrefillKernelResources &primary,
        DensePrefillKernelResources &auxiliary)
    {
#define QUERY_DENSE_PREFILL_TILE(ID, BM, BN, WM, WN)                         \
    case ID:                                                                 \
        return visitDensePrefillStaging<CB, BN, WM * WN * 32>(             \
            staging, [&]<PrefillStagingSchedule Staging>() {               \
                return queryDensePrefillBK64Resources<                     \
                    CB, BM, BN, WM, WN, Staging>(                          \
                        canonical_kpart, primary, auxiliary);             \
            })

        switch (tile_id)
        {
            QUERY_DENSE_PREFILL_TILE(0, 64, 64, 2, 2);
            QUERY_DENSE_PREFILL_TILE(1, 64, 128, 2, 2);
            QUERY_DENSE_PREFILL_TILE(2, 64, 128, 4, 2);
            QUERY_DENSE_PREFILL_TILE(3, 64, 128, 2, 4);
            QUERY_DENSE_PREFILL_TILE(4, 128, 128, 4, 2);
            QUERY_DENSE_PREFILL_TILE(5, 128, 128, 4, 4);
        default:
            return false;
        }

#undef QUERY_DENSE_PREFILL_TILE
    }

    /** Query one Q4_0 BK256 specialization selected by the last probe. */
    template <int BM, int BN, int WM, int WN>
    bool queryDensePrefillBK256Resources(
        bool ordered_reducer,
        DensePrefillKernelResources &primary,
        DensePrefillKernelResources &auxiliary)
    {
        constexpr int threads = WM * WN * 32;
        constexpr int dynamic_shared =
            nativeVnniBK256SharedMemoryBytes<BM, BN>();
        bool ok = false;
        if (ordered_reducer)
        {
            auto kernel = nativeVnniTC_BK256<BM, BN, WM, WN, true>;
            if (cudaFuncSetAttribute(
                    kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    dynamic_shared) != cudaSuccess)
            {
                (void)cudaGetLastError();
                return false;
            }
            ok = queryDensePrefillKernelResources(
                kernel, threads, dynamic_shared, primary);
        }
        else
        {
            auto kernel = nativeVnniTC_BK256<BM, BN, WM, WN, false>;
            if (cudaFuncSetAttribute(
                    kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    dynamic_shared) != cudaSuccess)
            {
                (void)cudaGetLastError();
                return false;
            }
            ok = queryDensePrefillKernelResources(
                kernel, threads, dynamic_shared, primary);
        }
        auxiliary = {};
        return ok;
    }

    /** Resolve resources for the exact specialization published by a probe. */
    template <uint8_t CB>
    bool queryLastDensePrefillKernelResources(
        int n,
        int k,
        int cuda_device_id,
        DensePrefillKernelResources &primary,
        DensePrefillKernelResources &auxiliary)
    {
        if (g_last_launch_selection.used_bk256)
        {
            if constexpr (CB != 0)
            {
                return false;
            }
            else
            {
                CUDAPrefillContext_ context;
                context.device_id = cuda_device_id;
                int ordered_reducer = 0;
                int k_partitions = 0;
                if (!cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
                        CB,
                        n,
                        k,
                        querySmCount(&context),
                        &ordered_reducer,
                        &k_partitions) ||
                    k_partitions <= 0)
                {
                    return false;
                }
                if (g_last_launch_selection.tile_id == -3)
                {
                    return queryDensePrefillBK256Resources<128, 64, 4, 2>(
                        ordered_reducer != 0, primary, auxiliary);
                }
                if (g_last_launch_selection.tile_id == -2)
                {
                    return queryDensePrefillBK256Resources<128, 128, 4, 4>(
                        ordered_reducer != 0, primary, auxiliary);
                }
                return false;
            }
        }

        return queryDensePrefillTileResources<CB>(
            g_last_launch_selection.tile_id,
            g_last_launch_selection.used_canonical_kpart != 0,
            g_last_launch_selection.staging,
            primary,
            auxiliary);
    }

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

        if (g_force_canonical_kpart)
            return Q40PrefillRoute::Generic;
        if (g_bk256_force_mode > 0)
            return bk256_geometry();
        if (isDensePrefillTileId(g_force_tile_id))
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

    // =========================================================================
    // Unified prefill dispatch: single format-agnostic path for ALL codebooks.
    //
    // Dispatch priority:
    //   1. BK256 path (CB=0 only): large-K shapes where BK64 can't fill GPU
    //   2. Force-tile override (g_force_tile_id): for sweep benchmarks
    //   3. Total heuristic dispatch (choosePrefillTile with format complexity)
    //   4. Tile launch: full-K or exact public-M1 K partitioning
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
        llaminar2::cuda::generated::CUDADensePrefillOverlayConfig
            exact_overlay{};
        const DensePrefillOverlayStatus overlay_status =
            selectDensePrefillOverlay(CB, M, N, K, exact_overlay);
        if (overlay_status == DensePrefillOverlayStatus::Invalid)
            return false;
        const bool use_exact_overlay =
            overlay_status == DensePrefillOverlayStatus::Selected;
        const PrefillStagingSchedule staging = use_exact_overlay
            ? exact_overlay.staging : g_staging_schedule;
        if constexpr (CB != 0)
        {
            if (use_exact_overlay && exact_overlay.bk256)
                return false;
        }

        Q40PrefillRoute q40_route = Q40PrefillRoute::Generic;

        // ─── BK256 path (CB=0 only) ──────────────────────────────────
        // BK256 processes 256 K-elements per outer tile (4× fewer K-iterations
        // than BK64), benefiting K-heavy shapes like FFN_Down (K=18944).
        // Uses 1 block/SM occupancy, so only helps when BK64 can't fill the GPU.
        if constexpr (CB == 0)
        {
            if (use_exact_overlay && exact_overlay.bk256)
            {
                q40_route = exact_overlay.tile_id == -3
                                ? Q40PrefillRoute::BK256Narrow
                                : Q40PrefillRoute::BK256Wide;
            }
            else
            {
                q40_route = chooseQ40PrefillRoute(M, N, K, prefill_ctx);
            }
            if (q40_route == Q40PrefillRoute::BK256Narrow ||
                q40_route == Q40PrefillRoute::BK256Wide)
            {
                // BK256 is a different physical kernel family; never silently
                // ignore an explicitly selected BK64 staging schedule.
                if (staging != PrefillStagingSchedule::RegisterDecode)
                    return false;
                constexpr int sk = 1;
                if (q40_route == Q40PrefillRoute::BK256Narrow)
                {
                    recordLastLaunchSelection(
                        -3, sk, true, false, use_exact_overlay);
                    return launchNativeVNNITC_BK256<128, 64, 4, 2>(
                        d_A_int8, d_payload, d_scales, d_C_fp32,
                        d_scales_A_block, M, N, K, alpha, beta,
                        d_C_existing, d_bias, cuda_stream,
                        serial_m1_k_partitions,
                        serial_m1_uses_ordered_reducer);
                }

                recordLastLaunchSelection(
                    -2, sk, true, false, use_exact_overlay);
                return launchNativeVNNITC_BK256<128, 128, 4, 4>(
                    d_A_int8, d_payload, d_scales, d_C_fp32,
                    d_scales_A_block, M, N, K, alpha, beta,
                    d_C_existing, d_bias, cuda_stream,
                    serial_m1_k_partitions,
                    serial_m1_uses_ordered_reducer);
            }
        }

        // ─── Tile selection (3-tier priority) ─────────────────────────
        TileChoice tc;

        if (use_exact_overlay)
        {
            tc = {static_cast<TileId>(exact_overlay.tile_id)};
        }
        else if constexpr (CB == 0)
        {
            if (q40_route == Q40PrefillRoute::ProfiledT64x64)
            {
                tc = {TileId::T64x64_w2x2};
            }
            else if (isDensePrefillTileId(g_force_tile_id))
            {
                tc = {static_cast<TileId>(g_force_tile_id)};
            }
            else
            {
                constexpr FormatComplexity complexity =
                    getFormatComplexity(CB);
                tc = choosePrefillTile(
                    M, N, K, prefill_ctx, complexity, CB);
            }
        }
        else if (isDensePrefillTileId(g_force_tile_id))
        {
            // Force-tile: bypass everything for sweep benchmarks
            tc = {static_cast<TileId>(g_force_tile_id)};
        }
        else
        {
            constexpr FormatComplexity complexity = getFormatComplexity(CB);
            tc = choosePrefillTile(
                M, N, K, prefill_ctx, complexity, CB);
        }

        // Tile launch: output geometry is selectable; reduction geometry is
        // either one complete K walk or the exact public-M1 partition tree.
        const bool use_canonical_kpart = use_exact_overlay
                                             ? exact_overlay.canonical_kpart
                                             : g_force_canonical_kpart;
        if (!densePrefillTileIsResourceEligible<CB>(
                tc.tile, use_canonical_kpart, staging))
        {
            /*
             * An exact overlay or forced profiler identity is a contract and
             * cannot be silently rewritten. Generic policy, however, must be
             * total over unseen geometry, so project an ineligible heuristic
             * choice onto tile 2: the all-codebook inventory proves that tile
             * spill-free for direct and canonical arithmetic alike.
             */
            if (use_exact_overlay ||
                staging != PrefillStagingSchedule::RegisterDecode ||
                isDensePrefillTileId(g_force_tile_id))
            {
                return false;
            }
            tc = {TileId::T64x128_w4x2};
        }
#define DISPATCH_TILE(BM_, BN_, WM_, WN_)                                              \
    do                                                                                 \
    {                                                                                  \
        return visitDensePrefillStaging<CB, BN_, WM_ * WN_ * 32>(                       \
            staging, [&]<PrefillStagingSchedule Staging>() {                            \
        if (use_canonical_kpart)                                                       \
        {                                                                              \
            if (!serial_m1_uses_ordered_reducer ||                                    \
                serial_m1_k_partitions <= 1)                                          \
                return false;                                                          \
            recordLastLaunchSelection(                                                \
                static_cast<int>(tc.tile),                                            \
                serial_m1_k_partitions,                                               \
                false,                                                                \
                true,                                                                 \
                use_exact_overlay, Staging);                                          \
            return launchNativeVNNITC_BK64CanonicalKpart<                             \
                CB, BM_, BN_, WM_, WN_, Staging>(                                     \
                d_A_int8, d_payload, d_scales, d_mins, d_emins,                      \
                d_C_fp32, d_scales_A_block, d_sums_A_block,                           \
                M, N, K, alpha, beta, d_C_existing, d_bias, cuda_stream,              \
                prefill_ctx, serial_m1_k_partitions);                                 \
        }                                                                              \
        recordLastLaunchSelection(                                                    \
            static_cast<int>(tc.tile), 1, false, false, use_exact_overlay, Staging);  \
        return launchNativeVNNITC_BK64<CB, BM_, BN_, WM_, WN_, Staging>(              \
            d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32,                 \
            d_scales_A_block, d_sums_A_block, M, N, K, alpha, beta,                   \
            d_C_existing, d_bias, cuda_stream,                                        \
            serial_m1_k_partitions, serial_m1_uses_ordered_reducer);                  \
            });                                                                       \
    } while (0)

        switch (tc.tile)
        {
        case TileId::T64x64_w2x2:
            DISPATCH_TILE(64, 64, 2, 2);
        case TileId::T64x128_w2x2:
            DISPATCH_TILE(64, 128, 2, 2);
        case TileId::T64x128_w4x2:
            DISPATCH_TILE(64, 128, 4, 2);
        case TileId::T64x128_w2x4:
            DISPATCH_TILE(64, 128, 2, 4);
        case TileId::T128x128_w4x2:
            DISPATCH_TILE(128, 128, 4, 2);
        case TileId::T128x128_w4x4:
            DISPATCH_TILE(128, 128, 4, 4);
        case TileId::Count:
            break;
        }

#undef DISPATCH_TILE
        return false; // unreachable
    }
}

extern "C"
{
    bool cudaNativeVNNIPrefill_setStagingSchedule(PrefillStagingSchedule schedule)
    {
        if (!isKnownPrefillStagingSchedule(schedule))
            return false;
        g_staging_schedule = schedule;
        return true;
    }

    PrefillStagingSchedule cudaNativeVNNIPrefill_getStagingSchedule()
    {
        return g_staging_schedule;
    }

    PrefillStagingSchedule cudaNativeVNNIPrefill_getLastLaunchStagingSchedule()
    {
        return g_last_launch_selection.staging;
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

    /** Select the exact public-M1 ordered K-partition candidate for a sweep. */
    void cudaNativeVNNIPrefill_setCanonicalKPartitionMode(bool enabled)
    {
        g_force_canonical_kpart = enabled;
    }

    /** Return whether the exact public-M1 K-partition candidate is forced. */
    bool cudaNativeVNNIPrefill_getCanonicalKPartitionMode()
    {
        return g_force_canonical_kpart;
    }

    /** Enable or bypass installed exact cells for an isolated policy sweep. */
    void cudaNativeVNNIPrefill_setExactOverlayEnabled(bool enabled)
    {
        g_dense_prefill_exact_overlay_enabled = enabled;
    }

    /** Return whether ordinary production may consult installed exact cells. */
    bool cudaNativeVNNIPrefill_getExactOverlayEnabled()
    {
        return g_dense_prefill_exact_overlay_enabled;
    }

    void cudaNativeVNNIPrefill_getLastLaunchSelection(
        int *tile_id,
        int *k_partitions,
        int *used_bk256,
        int *used_canonical_kpart)
    {
        if (tile_id)
            *tile_id = g_last_launch_selection.tile_id;
        if (k_partitions)
            *k_partitions = g_last_launch_selection.k_partitions;
        if (used_bk256)
            *used_bk256 = g_last_launch_selection.used_bk256;
        if (used_canonical_kpart)
            *used_canonical_kpart =
                g_last_launch_selection.used_canonical_kpart;
    }

    /**
     * @brief Report whether the last launch came from an installed exact cell.
     *
     * The value is thread-local for the same reason as the remaining launch
     * diagnostics: sweep workers configure and inspect their own launch
     * transaction. Production PerfStats uses this provenance to distinguish a
     * measured overlay from the total generic policy.
     */
    int cudaNativeVNNIPrefill_getLastLaunchUsedExactOverlay()
    {
        return g_last_launch_selection.used_exact_overlay;
    }

    /**
     * @brief Query compiler resources for the exact last-launched prefill path.
     *
     * A caller first performs one untimed launch on its exact producer stream.
     * That launch publishes the physical tile/BK256/canonical-reducer identity
     * in thread-local state. This setup-only query then inspects precisely that
     * primary specialization and, when present, the ordered reduction kernel.
     * It performs no allocation, transfer, kernel launch, or synchronization.
     *
     * @return `true` when both the primary and any auxiliary specialization
     *         have valid compiler attributes and non-zero theoretical
     *         occupancy; `false` for stale or unsupported launch identity.
     */
    bool cudaNativeVNNIPrefill_queryLastLaunchResources(
        uint8_t codebook_id,
        int n,
        int k,
        int cuda_device_id,
        CUDADensePrefillKernelResources *primary,
        CUDADensePrefillKernelResources *auxiliary)
    {
        if (!primary || !auxiliary || n <= 0 || k <= 0 ||
            cuda_device_id < 0 ||
            cudaSetDevice(cuda_device_id) != cudaSuccess)
        {
            return false;
        }

        *primary = {};
        *auxiliary = {};
        bool queried = false;
#define QUERY_LAST_DENSE_PREFILL_CODEBOOK(CB)                                \
    case CB:                                                                  \
        queried = queryLastDensePrefillKernelResources<CB>(                  \
            n, k, cuda_device_id, *primary, *auxiliary);                      \
        break

        switch (codebook_id)
        {
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(0);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(4);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(5);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(6);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(7);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(8);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(9);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(10);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(11);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(12);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(13);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(14);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(15);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(16);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(17);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(19);
            QUERY_LAST_DENSE_PREFILL_CODEBOOK(
                llaminar2::kNativeVnniExpandedInt8MinCodebook);
        default:
            return false;
        }

#undef QUERY_LAST_DENSE_PREFILL_CODEBOOK

        return queried;
    }

    bool cudaNativeVNNIPrefill_queryCandidateResources(
        uint8_t codebook_id,
        int tile_id,
        int canonical_kpart,
        int ordered_bk256,
        int cuda_device_id,
        CUDADensePrefillKernelResources *primary,
        CUDADensePrefillKernelResources *auxiliary,
        PrefillStagingSchedule staging)
    {
        if (!primary || !auxiliary || canonical_kpart < 0 ||
            canonical_kpart > 1 || ordered_bk256 < 0 ||
            ordered_bk256 > 1 || !isKnownPrefillStagingSchedule(staging) ||
            cuda_device_id < 0 ||
            cudaSetDevice(cuda_device_id) != cudaSuccess)
        {
            return false;
        }
        *primary = {};
        *auxiliary = {};

        if (tile_id == -3 || tile_id == -2)
        {
            if (codebook_id != 0 || canonical_kpart != 0 ||
                staging != PrefillStagingSchedule::RegisterDecode)
                return false;
            if (tile_id == -3)
            {
                return queryDensePrefillBK256Resources<128, 64, 4, 2>(
                    ordered_bk256 != 0, *primary, *auxiliary);
            }
            return queryDensePrefillBK256Resources<128, 128, 4, 4>(
                ordered_bk256 != 0, *primary, *auxiliary);
        }
        if (!isDensePrefillTileId(tile_id) || ordered_bk256 != 0)
            return false;

        bool queried = false;
#define QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(CB)                           \
    case CB:                                                                  \
        queried = queryDensePrefillTileResources<CB>(                        \
            tile_id, canonical_kpart != 0, staging, *primary, *auxiliary);    \
        break

        switch (codebook_id)
        {
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(0);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(4);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(5);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(6);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(7);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(8);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(9);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(10);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(11);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(12);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(13);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(14);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(15);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(16);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(17);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(19);
            QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK(
                llaminar2::kNativeVnniExpandedInt8MinCodebook);
        default:
            return false;
        }

#undef QUERY_DENSE_PREFILL_CANDIDATE_CODEBOOK

        return queried;
    }

    // Force output-tile geometry for sweep benchmarks. K partitioning remains
    // bound to the public M=1 arithmetic contract.
    void cudaNativeVNNIPrefill_setForceTile(int tile_id)
    {
        g_force_tile_id = tile_id;
    }

    void cudaNativeVNNIPrefill_getForceTile(int *tile_id)
    {
        if (tile_id)
            *tile_id = g_force_tile_id;
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
        case TileId::Count:
            return 0;
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
        float *canonical_kpart_partials,
        size_t canonical_kpart_partials_bytes)
    {
        if (!ctx)
            return;
        ctx->workspace_canonical_kpart_partials = canonical_kpart_partials;
        ctx->workspace_canonical_kpart_partials_size =
            canonical_kpart_partials_bytes;
    }

    /**
     * @brief Plan prefill workspace using distinct decode and arithmetic formats.
     *
     * The execution codebook selects the physical decoder and any learned
     * prefill overlay. The arithmetic-policy codebook selects the serial-M1
     * reduction tree that every row must reproduce. They differ after a cold
     * CPU tier promotes a compact asymmetric matrix into codebook 23.
     */
    bool cudaNativeVNNIPrefill_getWorkspacePlanWithPolicy(
        uint8_t codebook_id,
        uint8_t arithmetic_policy_codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions)
    {
        if (canonical_kpart_partials_bytes)
            *canonical_kpart_partials_bytes = 0;
        if (planned_k_partitions)
            *planned_k_partitions = 1;
        if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isSupportedNativeVNNIPrefillCodebook(codebook_id))
            return false;
        if (!cudaNativeVNNIGemvTuned_supportsCodebook(
                arithmetic_policy_codebook_id))
            return false;

        llaminar2::cuda::generated::CUDADensePrefillOverlayConfig
            exact_overlay{};
        const DensePrefillOverlayStatus overlay_status =
            selectDensePrefillOverlay(
                codebook_id, M, N, K, exact_overlay);
        if (overlay_status == DensePrefillOverlayStatus::Invalid)
            return false;
        const bool needs_canonical_kpart =
            g_force_canonical_kpart ||
            (overlay_status == DensePrefillOverlayStatus::Selected &&
             exact_overlay.canonical_kpart);
        if (!needs_canonical_kpart)
            return true;
        return queryCanonicalPrefillWorkspaceForRows(
            arithmetic_policy_codebook_id,
            M,
            N,
            K,
            cuda_device_id,
            canonical_kpart_partials_bytes,
            planned_k_partitions);
    }

    /** @brief Plan a self-describing matrix whose policy equals its decoder. */
    bool cudaNativeVNNIPrefill_getWorkspacePlan(
        uint8_t codebook_id,
        int M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions)
    {
        return cudaNativeVNNIPrefill_getWorkspacePlanWithPolicy(
            codebook_id,
            codebook_id,
            M,
            N,
            K,
            cuda_device_id,
            canonical_kpart_partials_bytes,
            planned_k_partitions);
    }

    /** @brief Plan the maximum workspace for an explicit arithmetic policy. */
    bool cudaNativeVNNIPrefill_getWorkspaceEnvelopeWithPolicy(
        uint8_t codebook_id,
        uint8_t arithmetic_policy_codebook_id,
        int max_M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions,
        int *planned_rows)
    {
        if (canonical_kpart_partials_bytes)
            *canonical_kpart_partials_bytes = 0;
        if (planned_k_partitions)
            *planned_k_partitions = 1;
        if (planned_rows)
            *planned_rows = 0;
        if (max_M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isSupportedNativeVNNIPrefillCodebook(codebook_id))
            return false;
        if (!cudaNativeVNNIGemvTuned_supportsCodebook(
                arithmetic_policy_codebook_id))
            return false;

        int canonical_rows = 0;
        if (g_force_canonical_kpart)
        {
            canonical_rows = max_M;
        }
        else
        {
            const bool exact_overlay_controls_launch =
                g_dense_prefill_exact_overlay_enabled &&
                g_force_tile_id < 0 &&
                g_bk256_force_mode == 0;
            if (exact_overlay_controls_launch)
            {
                /*
                 * Generated cells are sparse in geometry and non-monotonic in
                 * launch family. Scan the complete installed policy rather
                 * than assuming the largest M represents every smaller replay.
                 * This occurs during graph/workspace planning and is cached by
                 * the C++ adapter; no scan occurs in captured execution.
                 */
                for (const auto &entry :
                     llaminar2::cuda::generated::kCUDADensePrefillOverlayEntries)
                {
                    if (entry.codebook != codebook_id ||
                        entry.n != N || entry.k != K ||
                        entry.m <= 1 || entry.m > max_M)
                    {
                        continue;
                    }
                    if (!isValidDensePrefillOverlayConfig(
                            codebook_id, entry.config))
                    {
                        return false;
                    }
                    if (entry.config.canonical_kpart)
                        canonical_rows = std::max(canonical_rows, entry.m);
                }
            }
        }

        if (canonical_rows == 0)
            return true;
        if (!queryCanonicalPrefillWorkspaceForRows(
                arithmetic_policy_codebook_id,
                canonical_rows,
                N,
                K,
                cuda_device_id,
                canonical_kpart_partials_bytes,
                planned_k_partitions))
        {
            return false;
        }
        if (planned_rows)
            *planned_rows = canonical_rows;
        return true;
    }

    /** @brief Plan a self-describing matrix whose policy equals its decoder. */
    bool cudaNativeVNNIPrefill_getWorkspaceEnvelope(
        uint8_t codebook_id,
        int max_M,
        int N,
        int K,
        int cuda_device_id,
        size_t *canonical_kpart_partials_bytes,
        int *planned_k_partitions,
        int *planned_rows)
    {
        return cudaNativeVNNIPrefill_getWorkspaceEnvelopeWithPolicy(
            codebook_id,
            codebook_id,
            max_M,
            N,
            K,
            cuda_device_id,
            canonical_kpart_partials_bytes,
            planned_k_partitions,
            planned_rows);
    }
} // extern "C"

// Profitability gate removed: NativeVNNI is now the only CUDA GEMM path.
// TC/CUTLASS expanded fallback has been sunset. All codebooks always use
// the NativeVNNI prefill kernel regardless of shape.

/**
 * @brief Execute NativeVNNI prefill with explicit decoder and arithmetic IDs.
 *
 * `codebook_id` describes the bytes read by the templated kernel, while
 * `arithmetic_policy_codebook_id` owns the serial-M1 split tree. Keeping both
 * values explicit prevents representation normalization during expert
 * migration from changing FP32 publication order.
 */
extern "C" bool cudaNativeVNNIPrefill_fp32_withPolicy(
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
    uint8_t arithmetic_policy_codebook_id,
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
    if (!cudaNativeVNNIGemvTuned_supportsCodebook(
            arithmetic_policy_codebook_id))
        return false;
    if (!isAmperePlus(cuda_device_id))
        return false;
    if (cudaSetDevice(cuda_device_id) != cudaSuccess)
        return false;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    int serial_m1_uses_ordered_reducer = 0;
    int serial_m1_k_partitions = 0;
    if (!cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
            arithmetic_policy_codebook_id,
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

    case llaminar2::kNativeVnniExpandedInt8MinCodebook:
        if (!d_mins || !d_sums_A_block)
            return false;
        ok = launchGenericPrefillBK64<
            llaminar2::kNativeVnniExpandedInt8MinCodebook>(
            d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32,
            d_scales_A_block, d_sums_A_block, M, N, K, alpha, beta,
            d_C_existing, d_bias, serial_m1_k_partitions,
            serial_m1_uses_ordered_reducer, cuda_stream, prefill_ctx);
        break;

    default:
        return false;
    }

    if (ok && llaminar2::PerfStatsCollector::isDomainEnabled("kernel"))
    {
        int tile_id = -1;
        int k_partitions = 1;
        int used_bk256 = 0;
        int used_canonical_kpart = 0;
        int used_exact_overlay = 0;
        cudaNativeVNNIPrefill_getLastLaunchSelection(
            &tile_id,
            &k_partitions,
            &used_bk256,
            &used_canonical_kpart);
        used_exact_overlay =
            cudaNativeVNNIPrefill_getLastLaunchUsedExactOverlay();
        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "cuda_native_vnni_prefill_calls",
            1.0,
            "gemm",
            "cuda:" + std::to_string(cuda_device_id),
            llaminar2::PerfStatsCollector::Tags{
                {"codebook", std::to_string(static_cast<int>(codebook_id))},
                {"arithmetic_policy_codebook",
                 std::to_string(
                     static_cast<int>(arithmetic_policy_codebook_id))},
                {"m", std::to_string(M)},
                {"n", std::to_string(N)},
                {"k", std::to_string(K)},
                {"tile_id", std::to_string(tile_id)},
                {"k_partitions", std::to_string(k_partitions)},
                {"bk256", used_bk256 ? "1" : "0"},
                {"canonical_kpart", used_canonical_kpart ? "1" : "0"},
                {"staging_schedule", std::to_string(static_cast<int>(
                    cudaNativeVNNIPrefill_getLastLaunchStagingSchedule()))},
                {"exact_overlay", used_exact_overlay ? "1" : "0"},
                {"sums_a", d_sums_A_block ? "1" : "0"}});
    }
    return ok;
}

/** @brief Execute a self-describing matrix with its own arithmetic policy. */
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
    return cudaNativeVNNIPrefill_fp32_withPolicy(
        d_A_int8,
        d_payload,
        d_scales,
        d_mins,
        d_emins,
        d_C_fp32,
        d_scales_A_block,
        d_sums_A_block,
        M,
        N,
        K,
        alpha,
        beta,
        d_C_existing,
        d_bias,
        codebook_id,
        codebook_id,
        cuda_device_id,
        stream,
        prefill_ctx);
}
