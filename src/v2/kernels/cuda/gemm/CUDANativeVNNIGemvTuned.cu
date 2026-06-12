/**
 * @file CUDANativeVNNIGemvTuned.cu
 * @brief Tuned CUDA native-vnni GEMV kernels for decode (M=1).
 *
 * Three shape-specific kernel families:
 *   - **Wide**:     N >> K  (LM_Head only, N/K >= 44) — TILE_N columns per CTA, shared-memory A broadcast
 *   - **KPar**:     default (Attention, FFN_Up, FFN_Down) — K-split across CTAs,
 *                    two-phase reduction via partials buffer (no atomicAdd contention)
 *   - **Direct**:   Small N ≤ 512 (KV projections) — one thread per column, smem A broadcast
 *
 *   KPar Hybrid Reduction:
 *   Small partials (≤256KB): Two-phase — each CTA writes to d_partials[split_idx * N + n],
 *     then a reduce kernel sums partials → d_C[n].  Eliminates atomic contention.
 *   Large partials (>256KB):  Atomic — memset d_C, atomicAdd accumulation, then epilogue.
 *     Avoids the memory traffic overhead of the partials buffer.
 *   Threshold chosen empirically: 3B_Attn (128KB) gains +8.7%, 7B_Attn (401KB) loses -5.5%.
 *
 * All kernels decode the compact native payload inline and use dp4a for INT8 dot products.
 * The A (activation) vector is cached in shared memory once per K-block, eliminating
 * redundant global memory reads that plague the naive one-thread-per-column approach.
 *
 * Dispatch is tuned from the automated native-vnni sweep harness across all supported
 * codebooks and representative Qwen decode shapes.
 */

#include "kernels/cuda/gemm/CUDANativeVNNIDecodeCommon.cuh"
#include "kernels/cuda/gemm/CUDADeviceWorkspace.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <mutex>

// =====================================================================
// Per-device GEMV context — replaces static getSmCount() and
// getKparPartials(). Owned by KernelFactory, one per CUDA device.
// =====================================================================
struct CUDAGemvContext_
{
    int sm_count = 0;
    int device_id = -1;

    // KPAR two-phase reduction partials buffer
    float *kpar_partials = nullptr;
    size_t kpar_capacity = 0; // in floats
};

static int querySmCount(CUDAGemvContext_ *ctx)
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

static float *getKparPartials(CUDAGemvContext_ *ctx, size_t num_floats)
{
    if (ctx->kpar_partials && ctx->kpar_capacity >= num_floats)
        return ctx->kpar_partials;
    return nullptr;
}

// =====================================================================
// Row-major weight layout for row-parallel GEMV
//
// Column-major layout (current) is coalesced for column-parallel KPAR,
// but row-parallel (grid=N, one block per output row) needs row-major
// for coalesced K-dimension reads.  The data is created lazily on first
// ROWPAR dispatch and stored on CUDAPackedWeights — freed automatically
// when the weight tensor is destroyed.
//
// Memory cost: ~1× the weight data (about 1.4 GB for 7B Q4_0).
// Controlled by LLAMINAR_CUDA_GEMV_ROWPAR env variable.
// =====================================================================
struct CUDARowMajorWeights_
{
    uint8_t *d_payload = nullptr;
    uint16_t *d_scales = nullptr;
    uint16_t *d_mins = nullptr;
    uint32_t *d_emins = nullptr;
    int N = 0;
    int K_blocks = 0;
    int device_id = -1;
};

namespace
{
    static constexpr int BLOCK_K = 32;

    // =====================================================================
    // Sweep override — when active, dispatchCodebook uses these params
    // instead of classifyShape / selectKparTuning
    // =====================================================================
    struct SweepOverride
    {
        bool active = false;
        int kernel_family = -1; // 0=WIDE, 1=KPAR, 2=DIRECT
        int tile_n = 0;
        int cpt = 0;
        int target_waves = 0;
        int mkg = 0;
        int max_kb = 0;
        int force_two_phase = 0; // 0=auto, 1=force 2-phase, 2=force atomic
    };
    static SweepOverride g_sweep;

    // =====================================================================
    // Shape classifier — tuned from the native-vnni sweep results.
    // A slightly stricter wide threshold keeps borderline LM-head TP cases on
    // the KPAR path, which wins more consistently across compressed formats.
    // =====================================================================
    enum class NativeGemvShape
    {
        WIDE,
        KPAR,
        DIRECT,
        ROWPAR
    };

#include "kernels/cuda/gemm/CUDANativeVNNIGemvDispatchHeuristicGenerated.inc"

    [[maybe_unused]] static __host__ NativeGemvShape classifyShape(int N, int K)
    {
        if (N <= 512)
            return NativeGemvShape::DIRECT;
        if (N >= 44 * K)
            return NativeGemvShape::WIDE;
        return NativeGemvShape::KPAR;
    }

    // =====================================================================
    // Row-major weight layout for row-parallel GEMV
    // col_major[blk * N + n] → row_major[n * K_blocks + blk]
    // Each thread transposes one (n, blk) block by copying PB bytes.
    template <int PB>
    __global__ void transpose_blocks_kernel(
        const uint8_t *__restrict__ src,
        uint8_t *__restrict__ dst,
        int N, int K_blocks)
    {
        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total = N * K_blocks;
        if (idx >= total)
            return;

        const int n = idx / K_blocks;
        const int blk = idx % K_blocks;

        const uint8_t *s = src + (static_cast<size_t>(blk) * N + n) * PB;
        uint8_t *d = dst + (static_cast<size_t>(n) * K_blocks + blk) * PB;

        // Use typed copies for common payload sizes
        if constexpr (PB == 16)
        {
            *reinterpret_cast<int4 *>(d) = *reinterpret_cast<const int4 *>(s);
        }
        else if constexpr (PB == 2)
        {
            *reinterpret_cast<uint16_t *>(d) = *reinterpret_cast<const uint16_t *>(s);
        }
        else if constexpr (PB == 4)
        {
            *reinterpret_cast<uint32_t *>(d) = *reinterpret_cast<const uint32_t *>(s);
        }
        else
        {
            for (int i = 0; i < PB; ++i)
                d[i] = s[i];
        }
    }

    // Transpose a column-major buffer to row-major and return device pointer.
    // Returns nullptr on failure.
    template <typename T, int ELEM_BYTES>
    static T *transposeBuffer(const T *d_col, int N, int K_blocks, cudaStream_t stream)
    {
        const size_t total_elements = static_cast<size_t>(N) * K_blocks;
        const size_t total_bytes = total_elements * ELEM_BYTES;

        T *d_row = nullptr;
        if (cudaMalloc(&d_row, total_bytes) != cudaSuccess)
        {
            cudaGetLastError(); // Clear sticky error so KPAR fallback works
            return nullptr;
        }

        const int threads = 256;
        const int blocks = (static_cast<int>(total_elements) + threads - 1) / threads;
        transpose_blocks_kernel<ELEM_BYTES><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t *>(d_col),
            reinterpret_cast<uint8_t *>(d_row),
            N, K_blocks);

        if (cudaGetLastError() != cudaSuccess)
        {
            cudaFree(d_row);
            return nullptr;
        }
        return d_row;
    }

    // Row-parallel GEMV is enabled by default (fewer kernel launches = faster graph replay).
    // Set LLAMINAR_CUDA_GEMV_ROWPAR=0 to disable (e.g., if VRAM is tight).
    static bool isRowParEnabled()
    {
        static int enabled = -1;
        if (enabled < 0)
        {
            enabled = llaminar2::debugEnv().gemm.cuda_gemv_rowpar ? 1 : 0;
        }
        return enabled == 1;
    }

    static const char *shapeName(NativeGemvShape shape)
    {
        switch (shape)
        {
        case NativeGemvShape::WIDE:
            return "wide";
        case NativeGemvShape::KPAR:
            return "kpar";
        case NativeGemvShape::DIRECT:
            return "direct";
        case NativeGemvShape::ROWPAR:
            return "rowpar";
        }
        return "unknown";
    }

    template <uint8_t CB>
    static void recordGemvDispatch(
        NativeGemvShape shape,
        const GeneratedDispatchTuning &tuning,
        int N,
        int K,
        bool rowmajor_available,
        int cuda_device_id)
    {
        if (!llaminar2::PerfStatsCollector::isEnabled())
            return;

        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "cuda_native_vnni_gemv_dispatch",
            1.0,
            "decode",
            "cuda:" + std::to_string(cuda_device_id),
            llaminar2::PerfStatsCollector::Tags{
                {"codebook", std::to_string(static_cast<int>(CB))},
                {"n", std::to_string(N)},
                {"k", std::to_string(K)},
                {"route", shapeName(shape)},
                {"tile_n", std::to_string(tuning.tile_n)},
                {"cpt", std::to_string(tuning.cpt)},
                {"target_waves", std::to_string(tuning.target_waves)},
                {"mkg", std::to_string(tuning.mkg)},
                {"max_kb", std::to_string(tuning.max_kb)},
                {"force_two_phase", std::to_string(tuning.force_two_phase)},
                {"rowmajor_available", rowmajor_available ? "true" : "false"}});
    }

    // =====================================================================
    // K-split heuristic for KPar path
    // =====================================================================
    static int selectKSplit(int grid_n, int k_groups, int num_sms,
                            int target_waves, int min_kgroups_per_cta)
    {
        if (min_kgroups_per_cta <= 0)
            min_kgroups_per_cta = 2;
        int target_blocks = target_waves * num_sms;
        int kb = std::max(2, (target_blocks + grid_n - 1) / grid_n);
        int kb_max = std::max(2, k_groups / min_kgroups_per_cta);
        kb = std::min(kb, kb_max);

        // Snap to nearest factor of k_groups for even splits.
        // Find both nearest factor below and above, then pick the one
        // whose total block count is closer to the target.
        if (k_groups % kb != 0)
        {
            int best_lo = -1, best_hi = -1;
            for (int d = 1; d < kb; ++d)
                if (kb - d >= 2 && k_groups % (kb - d) == 0)
                {
                    best_lo = kb - d;
                    break;
                }
            for (int d = 1; kb + d <= kb_max; ++d)
                if (k_groups % (kb + d) == 0)
                {
                    best_hi = kb + d;
                    break;
                }

            if (best_lo > 0 && best_hi > 0)
            {
                int lo_dist = std::abs(grid_n * best_lo - target_blocks);
                int hi_dist = std::abs(grid_n * best_hi - target_blocks);
                kb = (hi_dist < lo_dist) ? best_hi : best_lo;
            }
            else if (best_lo > 0)
                kb = best_lo;
            else if (best_hi > 0)
                kb = best_hi;
        }
        return std::max(1, kb);
    }

    // =====================================================================
    // Kernel family 1: WIDE — N >> K (LM_Head, FFN_Up)
    //
    // Each CTA processes TILE_N output columns.
    // Each thread owns CPT consecutive columns.
    // A vector cached in shared memory per K-block (32 bytes = 8 int32_t).
    // Each thread decodes its weight payload block and dp4a against shared A.
    // =====================================================================
    template <int TILE_N, int CPT, uint8_t CB>
    __global__ void nativeVnniGemv_wide(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int N, int K,
        float alpha, float beta,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias)
    {
        const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
        if (n_base >= N)
            return;

        const int blocks_per_row = K / BLOCK_K;

        // Shared memory: 8 int32_t = 32 bytes for the A activation block
        __shared__ int32_t smem_A[8];

        float acc[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[c] = 0.0f;

        for (int blk = 0; blk < blocks_per_row; ++blk)
        {
            // Cooperative load of A block into shared memory (only first 8 threads)
            if (threadIdx.x < 8)
            {
                smem_A[threadIdx.x] = *reinterpret_cast<const int32_t *>(
                    d_A_int8 + blk * BLOCK_K + threadIdx.x * 4);
            }
            __syncthreads();

            const float scale_a = d_scales_A[blk];

// Process CPT columns
#pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n >= N)
                    break;

                const size_t linear = static_cast<size_t>(blk) * N + n;
                const uint8_t *payload = d_payload + linear *
                                                         llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CB>();

                int32_t packed_groups[8];
                llaminar2::cuda_native_vnni::decode_groups<CB>(payload, packed_groups);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
                {
                    int dot_lo = 0, dot_hi = 0;
                    int sum_lo = 0, sum_hi = 0;

#pragma unroll
                    for (int g = 0; g < 4; ++g)
                    {
                        dot_lo = __dp4a(smem_A[g], packed_groups[g], dot_lo);
                        dot_hi = __dp4a(smem_A[g + 4], packed_groups[g + 4], dot_hi);
                        sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[g]);
                        sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[g + 4]);
                    }

                    const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    const float scale_hi = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                    acc[c] += scale_a * (scale_lo * static_cast<float>(dot_lo) + scale_hi * static_cast<float>(dot_hi));

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                    {
                        const uint32_t emin = d_emins ? d_emins[linear] : 0u;
                        const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                        const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                        acc[c] += scale_a * (min_lo * static_cast<float>(sum_lo) + min_hi * static_cast<float>(sum_hi));
                    }

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
                    {
                        constexpr float IQ1S_DELTA = 0.125f;
                        const uint8_t qh0 = payload[4];
                        const uint8_t qh1 = payload[5];
                        const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[0]) + llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[1]);
                        const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[2]) + llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[3]);
                        const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[4]) + llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[5]);
                        const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[6]) + llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[7]);
                        const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        acc[c] += scale_a * ((d0 * static_cast<float>(sg0) + d1 * static_cast<float>(sg1)) * scale_lo +
                                             (d2 * static_cast<float>(sg2) + d3 * static_cast<float>(sg3)) * scale_hi);
                    }
                }
                else
                {
                    int dot = 0;
                    int sum_a = 0;
#pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        dot = __dp4a(smem_A[g], packed_groups[g], dot);
                        sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(smem_A[g]);
                    }

                    const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    acc[c] += scale_a * scale_b * static_cast<float>(dot);

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                    {
                        const float min_b = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                        acc[c] += scale_a * min_b * static_cast<float>(sum_a);
                    }
                }
            }

            __syncthreads(); // Ensure smem_A is not overwritten before all threads finish
        }

// Write output with alpha/beta/bias
#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                float out = alpha * acc[c];
                if (beta != 0.0f && d_C_existing)
                    out += beta * d_C_existing[n];
                if (d_bias)
                    out += d_bias[n];
                d_C[n] = out;
            }
        }
    }

    // =====================================================================
    // Kernel family 2: KPAR — K ≥ N (Attention, FFN_Down)
    //
    // K-dimension split across gridDim.y CTAs.
    // Each CTA processes a slice of K-blocks for CPT columns.
    // A vector cached in shared memory per K-block.
    //
    // TWO_PHASE=true:  writes partials to d_C[split_idx * N + n] (no atomics)
    // TWO_PHASE=false: atomic accumulation to d_C[n] (legacy fallback)
    // =====================================================================
    template <int TILE_N, int CPT, uint8_t CB, bool TWO_PHASE = false>
    __global__ void nativeVnniGemv_kpar(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int N, int K, int kb,
        float alpha)
    {
        const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
        const int split_idx = blockIdx.y;
        if (n_base >= N)
            return;

        const int blocks_per_row = K / BLOCK_K;
        const int blocks_per_split = (blocks_per_row + kb - 1) / kb;
        const int blk_begin = split_idx * blocks_per_split;
        const int blk_end = min(blocks_per_row, blk_begin + blocks_per_split);
        if (blk_begin >= blocks_per_row)
            return;

        float acc[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[c] = 0.0f;

        for (int blk = blk_begin; blk < blk_end; ++blk)
        {
            // Load A vector directly from global memory.  All threads in the
            // block read the same 32-byte activation chunk; the L1 cache
            // broadcasts it after the first warp's miss.  This eliminates the
            // previous __shared__ smem_A[8] + two __syncthreads() barriers per
            // k-block iteration, allowing warps to pipeline across iterations
            // independently — critical for low-work-per-CTA shapes (e.g. 3584×3584
            // attn with only 4 k-blocks/CTA where barriers were 78% L1TEX-stalled).
            //
            // Use 2 × int4 (128-bit) loads instead of 8 × int32 (32-bit) to
            // reduce instruction count from 8 LDG.E to 2 LDG.E.128 and let the
            // compiler allocate into consecutive register quads for ILP.
            int32_t a_vals[8];
            {
                const int4 *a_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + blk * BLOCK_K);
                const int4 a_lo = a_ptr128[0];
                const int4 a_hi = a_ptr128[1];
                a_vals[0] = a_lo.x;
                a_vals[1] = a_lo.y;
                a_vals[2] = a_lo.z;
                a_vals[3] = a_lo.w;
                a_vals[4] = a_hi.x;
                a_vals[5] = a_hi.y;
                a_vals[6] = a_hi.z;
                a_vals[7] = a_hi.w;
            }

            const float scale_a = d_scales_A[blk];

#pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n >= N)
                    break;

                const size_t linear = static_cast<size_t>(blk) * N + n;
                const uint8_t *payload = d_payload + linear *
                                                         llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CB>();

                int32_t packed_groups[8];
                // Always use vectorized decode: fewer instructions (e.g. 2 int4
                // loads vs 8 int32 for Q8_0), same DRAM traffic.  Previously
                // gated on TWO_PHASE but the choice is orthogonal to output path.
                llaminar2::cuda_native_vnni::decode_groups_vec<CB>(payload, packed_groups);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
                {
                    int dot_lo = 0, dot_hi = 0;
                    int sum_lo = 0, sum_hi = 0;

#pragma unroll
                    for (int g = 0; g < 4; ++g)
                    {
                        dot_lo = __dp4a(a_vals[g], packed_groups[g], dot_lo);
                        dot_hi = __dp4a(a_vals[g + 4], packed_groups[g + 4], dot_hi);
                        sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[g]);
                        sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[g + 4]);
                    }

                    const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    const float scale_hi = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                    acc[c] += scale_a * (scale_lo * static_cast<float>(dot_lo) + scale_hi * static_cast<float>(dot_hi));

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                    {
                        const uint32_t emin = d_emins ? d_emins[linear] : 0u;
                        const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                        const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                        acc[c] += scale_a * (min_lo * static_cast<float>(sum_lo) + min_hi * static_cast<float>(sum_hi));
                    }

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
                    {
                        constexpr float IQ1S_DELTA = 0.125f;
                        const uint8_t qh0 = payload[4];
                        const uint8_t qh1 = payload[5];
                        const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[0]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[1]);
                        const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[2]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[3]);
                        const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[4]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[5]);
                        const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[6]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[7]);
                        const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        acc[c] += scale_a * ((d0 * static_cast<float>(sg0) + d1 * static_cast<float>(sg1)) * scale_lo +
                                             (d2 * static_cast<float>(sg2) + d3 * static_cast<float>(sg3)) * scale_hi);
                    }
                }
                else
                {
                    int dot = 0;
                    int sum_a = 0;
#pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        dot = __dp4a(a_vals[g], packed_groups[g], dot);
                        sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[g]);
                    }

                    const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    acc[c] += scale_a * scale_b * static_cast<float>(dot);

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                    {
                        const float min_b = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                        acc[c] += scale_a * min_b * static_cast<float>(sum_a);
                    }
                }
            }
        }

// Output: two-phase (direct write to partials) or atomic fallback
#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                if constexpr (TWO_PHASE)
                    d_C[split_idx * N + n] = alpha * acc[c];
                else
                    atomicAdd(&d_C[n], alpha * acc[c]);
            }
        }
    }

    template <int TILE_N, int CPT, uint8_t CB, bool TWO_PHASE = false>
    __global__ void nativeVnniGemv_kpar_m2(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int N, int K, int kb,
        float alpha)
    {
        const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
        const int split_idx = blockIdx.y;
        if (n_base >= N)
            return;

        const int blocks_per_row = K / BLOCK_K;
        const int blocks_per_split = (blocks_per_row + kb - 1) / kb;
        const int blk_begin = split_idx * blocks_per_split;
        const int blk_end = min(blocks_per_row, blk_begin + blocks_per_split);
        if (blk_begin >= blocks_per_row)
            return;

        float acc0[CPT];
        float acc1[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            acc0[c] = 0.0f;
            acc1[c] = 0.0f;
        }

        for (int blk = blk_begin; blk < blk_end; ++blk)
        {
            int32_t a0_vals[8];
            int32_t a1_vals[8];
            {
                const int4 *a0_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + blk * BLOCK_K);
                const int4 *a1_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + K + blk * BLOCK_K);
                const int4 a0_lo = a0_ptr128[0];
                const int4 a0_hi = a0_ptr128[1];
                const int4 a1_lo = a1_ptr128[0];
                const int4 a1_hi = a1_ptr128[1];
                a0_vals[0] = a0_lo.x;
                a0_vals[1] = a0_lo.y;
                a0_vals[2] = a0_lo.z;
                a0_vals[3] = a0_lo.w;
                a0_vals[4] = a0_hi.x;
                a0_vals[5] = a0_hi.y;
                a0_vals[6] = a0_hi.z;
                a0_vals[7] = a0_hi.w;
                a1_vals[0] = a1_lo.x;
                a1_vals[1] = a1_lo.y;
                a1_vals[2] = a1_lo.z;
                a1_vals[3] = a1_lo.w;
                a1_vals[4] = a1_hi.x;
                a1_vals[5] = a1_hi.y;
                a1_vals[6] = a1_hi.z;
                a1_vals[7] = a1_hi.w;
            }

            const float scale_a0 = d_scales_A[blk];
            const float scale_a1 = d_scales_A[blocks_per_row + blk];

#pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n >= N)
                    break;

                const size_t linear = static_cast<size_t>(blk) * N + n;
                const uint8_t *payload = d_payload + linear *
                                                         llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CB>();

                int32_t packed_groups[8];
                llaminar2::cuda_native_vnni::decode_groups_vec<CB>(payload, packed_groups);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
                {
                    int dot0_lo = 0, dot0_hi = 0;
                    int dot1_lo = 0, dot1_hi = 0;
                    int sum0_lo = 0, sum0_hi = 0;
                    int sum1_lo = 0, sum1_hi = 0;

#pragma unroll
                    for (int g = 0; g < 4; ++g)
                    {
                        dot0_lo = __dp4a(a0_vals[g], packed_groups[g], dot0_lo);
                        dot0_hi = __dp4a(a0_vals[g + 4], packed_groups[g + 4], dot0_hi);
                        dot1_lo = __dp4a(a1_vals[g], packed_groups[g], dot1_lo);
                        dot1_hi = __dp4a(a1_vals[g + 4], packed_groups[g + 4], dot1_hi);
                        sum0_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[g]);
                        sum0_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[g + 4]);
                        sum1_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[g]);
                        sum1_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[g + 4]);
                    }

                    const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    const float scale_hi = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                    acc0[c] += scale_a0 * (scale_lo * static_cast<float>(dot0_lo) + scale_hi * static_cast<float>(dot0_hi));
                    acc1[c] += scale_a1 * (scale_lo * static_cast<float>(dot1_lo) + scale_hi * static_cast<float>(dot1_hi));

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                    {
                        const uint32_t emin = d_emins ? d_emins[linear] : 0u;
                        const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                        const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                        acc0[c] += scale_a0 * (min_lo * static_cast<float>(sum0_lo) + min_hi * static_cast<float>(sum0_hi));
                        acc1[c] += scale_a1 * (min_lo * static_cast<float>(sum1_lo) + min_hi * static_cast<float>(sum1_hi));
                    }

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
                    {
                        constexpr float IQ1S_DELTA = 0.125f;
                        const uint8_t qh0 = payload[4];
                        const uint8_t qh1 = payload[5];
                        const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const int sg0_0 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[0]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[1]);
                        const int sg0_1 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[2]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[3]);
                        const int sg0_2 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[4]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[5]);
                        const int sg0_3 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[6]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[7]);
                        const int sg1_0 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[0]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[1]);
                        const int sg1_1 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[2]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[3]);
                        const int sg1_2 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[4]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[5]);
                        const int sg1_3 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[6]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[7]);
                        acc0[c] += scale_a0 * ((d0 * static_cast<float>(sg0_0) + d1 * static_cast<float>(sg0_1)) * scale_lo +
                                               (d2 * static_cast<float>(sg0_2) + d3 * static_cast<float>(sg0_3)) * scale_hi);
                        acc1[c] += scale_a1 * ((d0 * static_cast<float>(sg1_0) + d1 * static_cast<float>(sg1_1)) * scale_lo +
                                               (d2 * static_cast<float>(sg1_2) + d3 * static_cast<float>(sg1_3)) * scale_hi);
                    }
                }
                else
                {
                    int dot0 = 0;
                    int dot1 = 0;
                    int sum0_a = 0;
                    int sum1_a = 0;
#pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        dot0 = __dp4a(a0_vals[g], packed_groups[g], dot0);
                        dot1 = __dp4a(a1_vals[g], packed_groups[g], dot1);
                        sum0_a += llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[g]);
                        sum1_a += llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[g]);
                    }

                    const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    acc0[c] += scale_a0 * scale_b * static_cast<float>(dot0);
                    acc1[c] += scale_a1 * scale_b * static_cast<float>(dot1);

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                    {
                        const float min_b = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                        acc0[c] += scale_a0 * min_b * static_cast<float>(sum0_a);
                        acc1[c] += scale_a1 * min_b * static_cast<float>(sum1_a);
                    }
                }
            }
        }

#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                if constexpr (TWO_PHASE)
                {
                    d_C[(split_idx * 2) * N + n] = alpha * acc0[c];
                    d_C[(split_idx * 2 + 1) * N + n] = alpha * acc1[c];
                }
                else
                {
                    atomicAdd(&d_C[n], alpha * acc0[c]);
                    atomicAdd(&d_C[N + n], alpha * acc1[c]);
                }
            }
        }
    }

    // =====================================================================
    // Kernel family 5: ROWPAR (Row-Parallel) — GEMV with one block per output row.
    //
    // Grid = N blocks, each block processes one output row with NWARPS warps.
    // Uses ROW-MAJOR weight layout: d_payload_rm[n * K_blocks + blk]
    // so that threads scanning K-blocks for the same row get coalesced reads.
    //
    // Eliminates inter-CTA reduction overhead (no partials buffer, no
    // memset, no epilogue kernel). Single kernel launch.
    //
    // NOTE: Q8_0 (CB==19) is excluded from ROWPAR because its 32-byte
    // payloads would require ~6.7 GB for the row-major cache, nearly
    // doubling VRAM. Q8_0 stays on KPAR which has naturally coalesced
    // column-major access.
    // =====================================================================
    template <int NWARPS, uint8_t CB>
    __global__ void nativeVnniGemv_rowpar(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload_rm,
        const uint16_t *__restrict__ d_scales_rm,
        const uint16_t *__restrict__ d_mins_rm,
        const uint32_t *__restrict__ d_emins_rm,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int N, int K,
        float alpha, float beta,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias)
    {
        constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
        const int n = blockIdx.x;
        if (n >= N)
            return;

        const int k_blocks = K / BLOCK_K;
        const int tid = threadIdx.x;
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;

        float acc = 0.0f;

        // Each thread processes K-blocks at stride blockDim.x
        for (int blk = tid; blk < k_blocks; blk += NWARPS * 32)
        {
            const float scale_a = d_scales_A[blk];

            // Load A data for this K-block from global memory (L2-cached across blocks)
            int32_t a_vals[8];
            {
                const int4 *a_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + blk * BLOCK_K);
                const int4 a_lo = a_ptr128[0];
                const int4 a_hi = a_ptr128[1];
                a_vals[0] = a_lo.x;
                a_vals[1] = a_lo.y;
                a_vals[2] = a_lo.z;
                a_vals[3] = a_lo.w;
                a_vals[4] = a_hi.x;
                a_vals[5] = a_hi.y;
                a_vals[6] = a_hi.z;
                a_vals[7] = a_hi.w;
            }

            // Row-major indexing: [n, blk]
            const size_t linear = static_cast<size_t>(n) * k_blocks + blk;
            const uint8_t *payload = d_payload_rm + linear * PB;

            int32_t packed_groups[8];
            llaminar2::cuda_native_vnni::decode_groups<CB>(payload, packed_groups);

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
            {
                int dot_lo = 0, dot_hi = 0;
                int sum_lo = 0, sum_hi = 0;
#pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    dot_lo = __dp4a(a_vals[g], packed_groups[g], dot_lo);
                    dot_hi = __dp4a(a_vals[g + 4], packed_groups[g + 4], dot_hi);
                    sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[g]);
                    sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[g + 4]);
                }

                const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales_rm[linear]);
                const float scale_hi = d_mins_rm ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins_rm[linear]) : 0.0f;
                acc += scale_a * (scale_lo * static_cast<float>(dot_lo) + scale_hi * static_cast<float>(dot_hi));

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                {
                    const uint32_t emin = d_emins_rm ? d_emins_rm[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                    const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                    acc += scale_a * (min_lo * static_cast<float>(sum_lo) + min_hi * static_cast<float>(sum_hi));
                }

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
                {
                    constexpr float IQ1S_DELTA = 0.125f;
                    const uint8_t qh0 = payload[4];
                    const uint8_t qh1 = payload[5];
                    const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[0]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[1]);
                    const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[2]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[3]);
                    const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[4]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[5]);
                    const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[6]) + llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[7]);
                    const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    acc += scale_a * ((d0 * static_cast<float>(sg0) + d1 * static_cast<float>(sg1)) * scale_lo +
                                      (d2 * static_cast<float>(sg2) + d3 * static_cast<float>(sg3)) * scale_hi);
                }
            }
            else
            {
                int dot = 0;
                int sum_a = 0;
#pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    dot = __dp4a(a_vals[g], packed_groups[g], dot);
                    sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[g]);
                }

                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales_rm[linear]);
                acc += scale_a * scale_b * static_cast<float>(dot);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                {
                    const float min_b = d_mins_rm ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins_rm[linear]) : 0.0f;
                    acc += scale_a * min_b * static_cast<float>(sum_a);
                }
            }
        }

        // Intra-warp reduction
        for (int mask = 16; mask > 0; mask >>= 1)
            acc += __shfl_xor_sync(0xFFFFFFFF, acc, mask);

        // Cross-warp reduction via shared memory
        __shared__ float reduce_smem[NWARPS];
        if (lane_id == 0)
            reduce_smem[warp_id] = acc;
        __syncthreads();

        if (tid == 0)
        {
            float sum = reduce_smem[0];
#pragma unroll
            for (int w = 1; w < NWARPS; ++w)
                sum += reduce_smem[w];

            float out = alpha * sum;
            if (beta != 0.0f && d_C_existing)
                out += beta * d_C_existing[n];
            if (d_bias)
                out += d_bias[n];
            d_C[n] = out;
        }
    }

    // ROWPAR verifier kernel for M=2. Each CTA owns one output column and
    // accumulates both activation rows while decoding the weight block once.
    template <int NWARPS, uint8_t CB>
    __global__ void nativeVnniGemv_rowpar_m2(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload_rm,
        const uint16_t *__restrict__ d_scales_rm,
        const uint16_t *__restrict__ d_mins_rm,
        const uint32_t *__restrict__ d_emins_rm,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int N, int K,
        float alpha, float beta,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias)
    {
        constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
        const int n = blockIdx.x;
        if (n >= N)
            return;

        const int k_blocks = K / BLOCK_K;
        const int tid = threadIdx.x;
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        for (int blk = tid; blk < k_blocks; blk += NWARPS * 32)
        {
            const int4 *a0_ptr128 = reinterpret_cast<const int4 *>(
                d_A_int8 + blk * BLOCK_K);
            const int4 *a1_ptr128 = reinterpret_cast<const int4 *>(
                d_A_int8 + K + blk * BLOCK_K);
            const int4 a0_lo = a0_ptr128[0];
            const int4 a0_hi = a0_ptr128[1];
            const int4 a1_lo = a1_ptr128[0];
            const int4 a1_hi = a1_ptr128[1];

            int32_t a0_vals[8];
            int32_t a1_vals[8];
            a0_vals[0] = a0_lo.x;
            a0_vals[1] = a0_lo.y;
            a0_vals[2] = a0_lo.z;
            a0_vals[3] = a0_lo.w;
            a0_vals[4] = a0_hi.x;
            a0_vals[5] = a0_hi.y;
            a0_vals[6] = a0_hi.z;
            a0_vals[7] = a0_hi.w;
            a1_vals[0] = a1_lo.x;
            a1_vals[1] = a1_lo.y;
            a1_vals[2] = a1_lo.z;
            a1_vals[3] = a1_lo.w;
            a1_vals[4] = a1_hi.x;
            a1_vals[5] = a1_hi.y;
            a1_vals[6] = a1_hi.z;
            a1_vals[7] = a1_hi.w;

            const size_t linear = static_cast<size_t>(n) * k_blocks + blk;
            const uint8_t *payload = d_payload_rm + linear * PB;

            int32_t packed_groups[8];
            llaminar2::cuda_native_vnni::decode_groups<CB>(payload, packed_groups);

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
            {
                int dot0_lo = 0, dot0_hi = 0;
                int dot1_lo = 0, dot1_hi = 0;
                int sum0_lo = 0, sum0_hi = 0;
                int sum1_lo = 0, sum1_hi = 0;
#pragma unroll
                for (int g = 0; g < 4; ++g)
                {
                    dot0_lo = __dp4a(a0_vals[g], packed_groups[g], dot0_lo);
                    dot0_hi = __dp4a(a0_vals[g + 4], packed_groups[g + 4], dot0_hi);
                    dot1_lo = __dp4a(a1_vals[g], packed_groups[g], dot1_lo);
                    dot1_hi = __dp4a(a1_vals[g + 4], packed_groups[g + 4], dot1_hi);
                    sum0_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[g]);
                    sum0_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[g + 4]);
                    sum1_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[g]);
                    sum1_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[g + 4]);
                }

                const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales_rm[linear]);
                const float scale_hi = d_mins_rm ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins_rm[linear]) : 0.0f;
                acc0 += d_scales_A[blk] * (scale_lo * static_cast<float>(dot0_lo) + scale_hi * static_cast<float>(dot0_hi));
                acc1 += d_scales_A[k_blocks + blk] * (scale_lo * static_cast<float>(dot1_lo) + scale_hi * static_cast<float>(dot1_hi));

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                {
                    const uint32_t emin = d_emins_rm ? d_emins_rm[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                    const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                    acc0 += d_scales_A[blk] * (min_lo * static_cast<float>(sum0_lo) + min_hi * static_cast<float>(sum0_hi));
                    acc1 += d_scales_A[k_blocks + blk] * (min_lo * static_cast<float>(sum1_lo) + min_hi * static_cast<float>(sum1_hi));
                }

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
                {
                    constexpr float IQ1S_DELTA = 0.125f;
                    const uint8_t qh0 = payload[4];
                    const uint8_t qh1 = payload[5];
                    const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                    const int sg0_0 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[0]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[1]);
                    const int sg0_1 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[2]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[3]);
                    const int sg0_2 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[4]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[5]);
                    const int sg0_3 = llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[6]) + llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[7]);
                    const int sg1_0 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[0]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[1]);
                    const int sg1_1 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[2]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[3]);
                    const int sg1_2 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[4]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[5]);
                    const int sg1_3 = llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[6]) + llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[7]);
                    acc0 += d_scales_A[blk] * ((d0 * static_cast<float>(sg0_0) + d1 * static_cast<float>(sg0_1)) * scale_lo +
                                               (d2 * static_cast<float>(sg0_2) + d3 * static_cast<float>(sg0_3)) * scale_hi);
                    acc1 += d_scales_A[k_blocks + blk] * ((d0 * static_cast<float>(sg1_0) + d1 * static_cast<float>(sg1_1)) * scale_lo +
                                                          (d2 * static_cast<float>(sg1_2) + d3 * static_cast<float>(sg1_3)) * scale_hi);
                }
            }
            else
            {
                int dot0 = 0;
                int dot1 = 0;
                int sum0_a = 0;
                int sum1_a = 0;
#pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    dot0 = __dp4a(a0_vals[g], packed_groups[g], dot0);
                    dot1 = __dp4a(a1_vals[g], packed_groups[g], dot1);
                    sum0_a += llaminar2::cuda_native_vnni::sum_packed_i8(a0_vals[g]);
                    sum1_a += llaminar2::cuda_native_vnni::sum_packed_i8(a1_vals[g]);
                }

                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales_rm[linear]);
                acc0 += d_scales_A[blk] * scale_b * static_cast<float>(dot0);
                acc1 += d_scales_A[k_blocks + blk] * scale_b * static_cast<float>(dot1);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                {
                    const float min_b = d_mins_rm ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins_rm[linear]) : 0.0f;
                    acc0 += d_scales_A[blk] * min_b * static_cast<float>(sum0_a);
                    acc1 += d_scales_A[k_blocks + blk] * min_b * static_cast<float>(sum1_a);
                }
            }
        }

        for (int mask = 16; mask > 0; mask >>= 1)
        {
            acc0 += __shfl_xor_sync(0xFFFFFFFF, acc0, mask);
            acc1 += __shfl_xor_sync(0xFFFFFFFF, acc1, mask);
        }

        __shared__ float reduce0[NWARPS];
        __shared__ float reduce1[NWARPS];
        if (lane_id == 0)
        {
            reduce0[warp_id] = acc0;
            reduce1[warp_id] = acc1;
        }
        __syncthreads();

        if (tid == 0)
        {
            float sum0 = reduce0[0];
            float sum1 = reduce1[0];
#pragma unroll
            for (int w = 1; w < NWARPS; ++w)
            {
                sum0 += reduce0[w];
                sum1 += reduce1[w];
            }

            float out0 = alpha * sum0;
            float out1 = alpha * sum1;
            if (beta != 0.0f && d_C_existing)
            {
                out0 += beta * d_C_existing[n];
                out1 += beta * d_C_existing[N + n];
            }
            if (d_bias)
            {
                out0 += d_bias[n];
                out1 += d_bias[n];
            }
            d_C[n] = out0;
            d_C[N + n] = out1;
        }
    }

    // ROWPAR verifier kernel for M=2..4. Each CTA owns one output column and
    // accumulates all activation rows while decoding the weight block once.
    template <int M, int NWARPS, uint8_t CB>
    __global__ void nativeVnniGemv_rowpar_small_m(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload_rm,
        const uint16_t *__restrict__ d_scales_rm,
        const uint16_t *__restrict__ d_mins_rm,
        const uint32_t *__restrict__ d_emins_rm,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int N, int K,
        float alpha, float beta,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
        const int n = blockIdx.x;
        if (n >= N)
            return;

        const int k_blocks = K / BLOCK_K;
        const int tid = threadIdx.x;
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;

        float acc[M];
#pragma unroll
        for (int row = 0; row < M; ++row)
            acc[row] = 0.0f;

        for (int blk = tid; blk < k_blocks; blk += NWARPS * 32)
        {
            int32_t a_vals[M][8];
#pragma unroll
            for (int row = 0; row < M; ++row)
            {
                const int4 *a_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + static_cast<size_t>(row) * K + blk * BLOCK_K);
                const int4 a_lo = a_ptr128[0];
                const int4 a_hi = a_ptr128[1];
                a_vals[row][0] = a_lo.x;
                a_vals[row][1] = a_lo.y;
                a_vals[row][2] = a_lo.z;
                a_vals[row][3] = a_lo.w;
                a_vals[row][4] = a_hi.x;
                a_vals[row][5] = a_hi.y;
                a_vals[row][6] = a_hi.z;
                a_vals[row][7] = a_hi.w;
            }

            const size_t linear = static_cast<size_t>(n) * k_blocks + blk;
            const uint8_t *payload = d_payload_rm + linear * PB;

            int32_t packed_groups[8];
            llaminar2::cuda_native_vnni::decode_groups<CB>(payload, packed_groups);

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
            {
                const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales_rm[linear]);
                const float scale_hi = d_mins_rm ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins_rm[linear]) : 0.0f;
                const uint32_t emin = d_emins_rm ? d_emins_rm[linear] : 0u;
                const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));

#pragma unroll
                for (int row = 0; row < M; ++row)
                {
                    int dot_lo = 0;
                    int dot_hi = 0;
                    int sum_lo = 0;
                    int sum_hi = 0;
#pragma unroll
                    for (int g = 0; g < 4; ++g)
                    {
                        dot_lo = __dp4a(a_vals[row][g], packed_groups[g], dot_lo);
                        dot_hi = __dp4a(a_vals[row][g + 4], packed_groups[g + 4], dot_hi);
                        sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][g]);
                        sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][g + 4]);
                    }

                    const float scale_a = d_scales_A[static_cast<size_t>(row) * k_blocks + blk];
                    acc[row] += scale_a * (scale_lo * static_cast<float>(dot_lo) +
                                           scale_hi * static_cast<float>(dot_hi));

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                    {
                        acc[row] += scale_a * (min_lo * static_cast<float>(sum_lo) +
                                               min_hi * static_cast<float>(sum_hi));
                    }

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
                    {
                        constexpr float IQ1S_DELTA = 0.125f;
                        const uint8_t qh0 = payload[4];
                        const uint8_t qh1 = payload[5];
                        const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                        const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][0]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][1]);
                        const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][2]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][3]);
                        const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][4]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][5]);
                        const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][6]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][7]);
                        acc[row] += scale_a * ((d0 * static_cast<float>(sg0) + d1 * static_cast<float>(sg1)) * scale_lo +
                                               (d2 * static_cast<float>(sg2) + d3 * static_cast<float>(sg3)) * scale_hi);
                    }
                }
            }
            else
            {
                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales_rm[linear]);
                const float min_b = d_mins_rm ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins_rm[linear]) : 0.0f;

#pragma unroll
                for (int row = 0; row < M; ++row)
                {
                    int dot = 0;
                    int sum_a = 0;
#pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        dot = __dp4a(a_vals[row][g], packed_groups[g], dot);
                        sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][g]);
                    }

                    const float scale_a = d_scales_A[static_cast<size_t>(row) * k_blocks + blk];
                    acc[row] += scale_a * scale_b * static_cast<float>(dot);

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                    {
                        acc[row] += scale_a * min_b * static_cast<float>(sum_a);
                    }
                }
            }
        }

#pragma unroll
        for (int row = 0; row < M; ++row)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                acc[row] += __shfl_xor_sync(0xFFFFFFFF, acc[row], mask);
        }

        __shared__ float reduce[M][NWARPS];
        if (lane_id == 0)
        {
#pragma unroll
            for (int row = 0; row < M; ++row)
                reduce[row][warp_id] = acc[row];
        }
        __syncthreads();

        if (tid == 0)
        {
#pragma unroll
            for (int row = 0; row < M; ++row)
            {
                float sum = reduce[row][0];
#pragma unroll
                for (int w = 1; w < NWARPS; ++w)
                    sum += reduce[row][w];

                const size_t out_idx = static_cast<size_t>(row) * N + n;
                float out = alpha * sum;
                if (beta != 0.0f && d_C_existing)
                    out += beta * d_C_existing[out_idx];
                if (d_bias)
                    out += d_bias[n];
                d_C[out_idx] = out;
            }
        }
    }

    // =====================================================================
    // Reduce kernel: sum kb partials per column + apply beta/bias
    //
    // Replaces both cudaMemsetAsync (no need to zero d_C) and the separate
    // epilogue kernel — 3 launches → 2 launches.
    //
    // Layout: d_partials[k * N + n] for k=0..kb-1
    // Adjacent threads read adjacent n values → coalesced for each k.
    // =====================================================================
    __global__ void nativeVnniGemv_reduce(
        const float *__restrict__ d_partials,
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int N, int kb, float beta)
    {
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N)
            return;

        float sum = 0.0f;
        for (int k = 0; k < kb; ++k)
            sum += d_partials[k * N + n];

        if (beta != 0.0f && d_C_existing)
            sum += beta * d_C_existing[n];
        if (d_bias)
            sum += d_bias[n];
        d_C[n] = sum;
    }

    __global__ void nativeVnniGemv_reduce_m2(
        const float *__restrict__ d_partials,
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int N, int kb, float beta)
    {
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N)
            return;

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        for (int k = 0; k < kb; ++k)
        {
            sum0 += d_partials[(k * 2) * N + n];
            sum1 += d_partials[(k * 2 + 1) * N + n];
        }

        if (beta != 0.0f && d_C_existing)
        {
            sum0 += beta * d_C_existing[n];
            sum1 += beta * d_C_existing[N + n];
        }
        if (d_bias)
        {
            sum0 += d_bias[n];
            sum1 += d_bias[n];
        }
        d_C[n] = sum0;
        d_C[N + n] = sum1;
    }

    // =====================================================================
    // Epilogue kernel: apply beta/bias after atomic KPAR accumulation
    //
    // Used by the atomic fallback path where d_C already has the accumulated
    // alpha * A * W sum via atomicAdd.  Applies beta * C_existing + bias.
    // =====================================================================
    __global__ void nativeVnniGemv_epilogue(
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int N, float beta)
    {
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N)
            return;
        if (beta != 0.0f && d_C_existing)
            d_C[n] += beta * d_C_existing[n];
        if (d_bias)
            d_C[n] += d_bias[n];
    }

    __global__ void nativeVnniGemv_epilogue_m2(
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int N, float beta)
    {
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N)
            return;
        if (beta != 0.0f && d_C_existing)
        {
            d_C[n] += beta * d_C_existing[n];
            d_C[N + n] += beta * d_C_existing[N + n];
        }
        if (d_bias)
        {
            d_C[n] += d_bias[n];
            d_C[N + n] += d_bias[n];
        }
    }

    template <int M, int TILE_N, int CPT, uint8_t CB, bool TWO_PHASE = false>
    __global__ void nativeVnniGemv_kpar_small_m(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int N, int K, int kb,
        float alpha)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
        const int split_idx = blockIdx.y;
        if (n_base >= N)
            return;

        const int blocks_per_row = K / BLOCK_K;
        const int blocks_per_split = (blocks_per_row + kb - 1) / kb;
        const int blk_begin = split_idx * blocks_per_split;
        const int blk_end = min(blocks_per_row, blk_begin + blocks_per_split);
        if (blk_begin >= blocks_per_row)
            return;

        float acc[M][CPT];
#pragma unroll
        for (int row = 0; row < M; ++row)
        {
#pragma unroll
            for (int c = 0; c < CPT; ++c)
                acc[row][c] = 0.0f;
        }

        for (int blk = blk_begin; blk < blk_end; ++blk)
        {
            int32_t a_vals[M][8];
#pragma unroll
            for (int row = 0; row < M; ++row)
            {
                const int4 *a_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + static_cast<size_t>(row) * K + blk * BLOCK_K);
                const int4 a_lo = a_ptr128[0];
                const int4 a_hi = a_ptr128[1];
                a_vals[row][0] = a_lo.x;
                a_vals[row][1] = a_lo.y;
                a_vals[row][2] = a_lo.z;
                a_vals[row][3] = a_lo.w;
                a_vals[row][4] = a_hi.x;
                a_vals[row][5] = a_hi.y;
                a_vals[row][6] = a_hi.z;
                a_vals[row][7] = a_hi.w;
            }

#pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n >= N)
                    break;

                const size_t linear = static_cast<size_t>(blk) * N + n;
                const uint8_t *payload = d_payload + linear *
                                                         llaminar2::cuda_native_vnni::payload_bytes_for_codebook<CB>();

                int32_t packed_groups[8];
                llaminar2::cuda_native_vnni::decode_groups_vec<CB>(payload, packed_groups);

                if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
                {
                    const float scale_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    const float scale_hi = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                    const uint32_t emin = d_emins ? d_emins[linear] : 0u;
                    const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                    const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));

#pragma unroll
                    for (int row = 0; row < M; ++row)
                    {
                        int dot_lo = 0;
                        int dot_hi = 0;
                        int sum_lo = 0;
                        int sum_hi = 0;
#pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            dot_lo = __dp4a(a_vals[row][g], packed_groups[g], dot_lo);
                            dot_hi = __dp4a(a_vals[row][g + 4], packed_groups[g + 4], dot_hi);
                            sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][g]);
                            sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][g + 4]);
                        }

                        const float scale_a = d_scales_A[static_cast<size_t>(row) * blocks_per_row + blk];
                        acc[row][c] += scale_a * (scale_lo * static_cast<float>(dot_lo) +
                                                  scale_hi * static_cast<float>(dot_hi));

                        if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                        {
                            acc[row][c] += scale_a * (min_lo * static_cast<float>(sum_lo) +
                                                      min_hi * static_cast<float>(sum_hi));
                        }

                        if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
                        {
                            constexpr float IQ1S_DELTA = 0.125f;
                            const uint8_t qh0 = payload[4];
                            const uint8_t qh1 = payload[5];
                            const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                            const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][0]) +
                                            llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][1]);
                            const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][2]) +
                                            llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][3]);
                            const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][4]) +
                                            llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][5]);
                            const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][6]) +
                                            llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][7]);
                            acc[row][c] += scale_a * ((d0 * static_cast<float>(sg0) + d1 * static_cast<float>(sg1)) * scale_lo +
                                                      (d2 * static_cast<float>(sg2) + d3 * static_cast<float>(sg3)) * scale_hi);
                        }
                    }
                }
                else
                {
                    const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales[linear]);
                    const float min_b = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;

#pragma unroll
                    for (int row = 0; row < M; ++row)
                    {
                        int dot = 0;
                        int sum_a = 0;
#pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            dot = __dp4a(a_vals[row][g], packed_groups[g], dot);
                            sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[row][g]);
                        }

                        const float scale_a = d_scales_A[static_cast<size_t>(row) * blocks_per_row + blk];
                        acc[row][c] += scale_a * scale_b * static_cast<float>(dot);

                        if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                        {
                            acc[row][c] += scale_a * min_b * static_cast<float>(sum_a);
                        }
                    }
                }
            }
        }

#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
#pragma unroll
                for (int row = 0; row < M; ++row)
                {
                    if constexpr (TWO_PHASE)
                    {
                        d_C[(static_cast<size_t>(split_idx) * M + row) * N + n] =
                            alpha * acc[row][c];
                    }
                    else
                    {
                        atomicAdd(&d_C[static_cast<size_t>(row) * N + n],
                                  alpha * acc[row][c]);
                    }
                }
            }
        }
    }

    template <int M>
    __global__ void nativeVnniGemv_reduce_small_m(
        const float *__restrict__ d_partials,
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int N, int kb, float beta)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N)
            return;

#pragma unroll
        for (int row = 0; row < M; ++row)
        {
            float sum = 0.0f;
            for (int k = 0; k < kb; ++k)
            {
                sum += d_partials[(static_cast<size_t>(k) * M + row) * N + n];
            }

            const size_t out_idx = static_cast<size_t>(row) * N + n;
            if (beta != 0.0f && d_C_existing)
                sum += beta * d_C_existing[out_idx];
            if (d_bias)
                sum += d_bias[n];
            d_C[out_idx] = sum;
        }
    }

    template <int M>
    __global__ void nativeVnniGemv_epilogue_small_m(
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int N, float beta)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (n >= N)
            return;

#pragma unroll
        for (int row = 0; row < M; ++row)
        {
            const size_t out_idx = static_cast<size_t>(row) * N + n;
            if (beta != 0.0f && d_C_existing)
                d_C[out_idx] += beta * d_C_existing[out_idx];
            if (d_bias)
                d_C[out_idx] += d_bias[n];
        }
    }

    // =====================================================================
    // KPAR tile profiles — different TILE_N × CPT × K-split combinations
    // =====================================================================
    enum class KparTile
    {
        T32_C1,
        T64_C1,
        T64_C2,
        T128_C1,
        T128_C2,
        T256_C2
    };

    enum class WideTile
    {
        T32_C1,
        T64_C1,
        T64_C2,
        T128_C1,
        T128_C2,
        T256_C2,
        T256_C4,
        T512_C4
    };

    struct KparTuning
    {
        KparTile tile;
        int target_waves;
        int min_kgroups_per_cta;
        int max_kb; // Hard cap on K-splits (0 = no cap)
    };

    struct WideTuning
    {
        WideTile tile;
    };

    template <int TILE_N, int CPT, uint8_t CB>
    bool sweepLaunchKpar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int target_waves, int min_kgroups_per_cta, int max_kb,
        int force_two_phase,
        CUDAGemvContext_ *gemv_ctx,
        int device_id, cudaStream_t stream);

    // Two-phase partials buffer threshold (bytes).
    // Shapes with partials ≤ this use two-phase (no atomics); larger use atomic.
    //
    // Empirical results (Q4_0, RTX 3090, mkg=4):
    //   128KB  3B_Attn:    +8.7% win (12.5% → 21.2%)
    //   136KB  0.5B_FFN_Up:+2.2% win
    //   401KB  7B_Attn:    -5.5% loss (atomic contention negligible, reduce overhead dominates)
    //   704KB  3B_FFN_Up:  -4.4% loss
    static constexpr size_t kTwoPhaseMaxBytes = 256 * 1024;

    // Select KPAR tuning profile.
    //
    // The sweep consistently favored the narrower 128x1 tile for the native-
    // payload path. More compressed codebooks benefit from higher CTA wave
    // pressure, while denser formats saturate with 8 target waves.
    template <uint8_t CB>
    static KparTuning selectKparTuning([[maybe_unused]] int N, [[maybe_unused]] int K)
    {
        constexpr int payload_bytes = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
        constexpr int target_waves = (payload_bytes <= 13) ? 16 : 8;
        return {KparTile::T128_C1, target_waves, 2, 0};
    }

    // Wide/native-direct shapes favor the simple 128x1 kernel for decode.
    [[maybe_unused]] static WideTuning selectWideTuning([[maybe_unused]] int N, [[maybe_unused]] int K)
    {
        return {WideTile::T128_C1};
    }

    // =====================================================================
    // Dispatch helpers — launch a specific codebook with the right kernel family
    // =====================================================================
    template <uint8_t CB>
    bool launchWide(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        cudaStream_t stream)
    {
        const auto t = selectWideTuning(N, K);

        switch (t.tile)
        {
        case WideTile::T32_C1:
        {
            const int grid_n = (N + 32 - 1) / 32;
            nativeVnniGemv_wide<32, 1, CB><<<grid_n, 32, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T64_C1:
        {
            const int grid_n = (N + 64 - 1) / 64;
            nativeVnniGemv_wide<64, 1, CB><<<grid_n, 64, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T64_C2:
        {
            const int grid_n = (N + 64 - 1) / 64;
            nativeVnniGemv_wide<64, 2, CB><<<grid_n, 32, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T128_C1:
        {
            const int grid_n = (N + 128 - 1) / 128;
            nativeVnniGemv_wide<128, 1, CB><<<grid_n, 128, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T128_C2:
        {
            const int grid_n = (N + 128 - 1) / 128;
            nativeVnniGemv_wide<128, 2, CB><<<grid_n, 64, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T256_C2:
        {
            const int grid_n = (N + 256 - 1) / 256;
            nativeVnniGemv_wide<256, 2, CB><<<grid_n, 128, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T256_C4:
        {
            const int grid_n = (N + 256 - 1) / 256;
            nativeVnniGemv_wide<256, 4, CB><<<grid_n, 64, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T512_C4:
        {
            const int grid_n = (N + 512 - 1) / 512;
            nativeVnniGemv_wide<512, 4, CB><<<grid_n, 128, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        }

        return cudaGetLastError() == cudaSuccess;
    }

    template <int TILE_N, int CPT, uint8_t CB>
    bool launchKparImpl(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int target_waves, int min_kgroups_per_cta, int max_kb,
        int force_two_phase,
        int device_id, cudaStream_t stream,
        CUDAGemvContext_ *gemv_ctx)
    {
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const int kb = selectKSplit(grid_n, k_groups, num_sms,
                                    target_waves, min_kgroups_per_cta);
        const bool deterministic = llaminar2::debugEnv().gemm.deterministic;
        const int kb_capped_auto = (max_kb > 0) ? std::min(kb, max_kb) : kb;
        const int kb_capped = deterministic ? 1 : kb_capped_auto;

        // Choose two-phase vs atomic based on partials buffer size.
        // Deterministic mode always uses two-phase to avoid atomicAdd non-determinism.
        const size_t partials_bytes = static_cast<size_t>(kb_capped) * N * sizeof(float);
        const bool use_two_phase = deterministic || (partials_bytes <= kTwoPhaseMaxBytes);

        dim3 grid(grid_n, kb_capped);

        if (use_two_phase)
        {
            // Two-phase: write partials, then reduce
            float *d_partials = getKparPartials(
                gemv_ctx, static_cast<size_t>(kb_capped) * N);
            if (!d_partials)
                return false;

            nativeVnniGemv_kpar<TILE_N, CPT, CB, true><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_partials, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            const int rblk = (N + 255) / 256;
            nativeVnniGemv_reduce<<<rblk, 256, 0, stream>>>(
                d_partials, d_C, d_C_existing, d_bias, N, kb_capped, beta);
        }
        else
        {
            // Atomic: zero output, accumulate via atomicAdd, then epilogue
            cudaMemsetAsync(d_C, 0, static_cast<size_t>(N) * sizeof(float), stream);

            nativeVnniGemv_kpar<TILE_N, CPT, CB, false><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            // Apply beta/bias if needed
            if ((beta != 0.0f && d_C_existing) || d_bias)
            {
                const int eblk = (N + 255) / 256;
                nativeVnniGemv_epilogue<<<eblk, 256, 0, stream>>>(
                    d_C, d_C_existing, d_bias, N, beta);
            }
        }

        return cudaGetLastError() == cudaSuccess;
    }

    template <int TILE_N, int CPT, uint8_t CB>
    bool launchKparM2Impl(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int target_waves, int min_kgroups_per_cta, int max_kb,
        int force_two_phase,
        int device_id, cudaStream_t stream,
        CUDAGemvContext_ *gemv_ctx)
    {
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const int kb = selectKSplit(grid_n, k_groups, num_sms,
                                    target_waves, min_kgroups_per_cta);
        const bool deterministic = llaminar2::debugEnv().gemm.deterministic;
        const int kb_capped_auto = (max_kb > 0) ? std::min(kb, max_kb) : kb;
        const int kb_capped = deterministic ? 1 : kb_capped_auto;

        const size_t partials_bytes = static_cast<size_t>(kb_capped) * 2 * N * sizeof(float);
        bool use_two_phase;
        if (deterministic)
            use_two_phase = true;
        else if (force_two_phase == 1)
            use_two_phase = true;
        else if (force_two_phase == 2)
            use_two_phase = false;
        else
            use_two_phase = (partials_bytes <= kTwoPhaseMaxBytes);

        dim3 grid(grid_n, kb_capped);

        if (use_two_phase)
        {
            float *d_partials = getKparPartials(
                gemv_ctx, static_cast<size_t>(kb_capped) * 2 * N);
            if (!d_partials)
                return false;

            nativeVnniGemv_kpar_m2<TILE_N, CPT, CB, true><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_partials, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            const int rblk = (N + 255) / 256;
            nativeVnniGemv_reduce_m2<<<rblk, 256, 0, stream>>>(
                d_partials, d_C, d_C_existing, d_bias, N, kb_capped, beta);
        }
        else
        {
            cudaMemsetAsync(d_C, 0, static_cast<size_t>(2) * N * sizeof(float), stream);

            nativeVnniGemv_kpar_m2<TILE_N, CPT, CB, false><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            if ((beta != 0.0f && d_C_existing) || d_bias)
            {
                const int eblk = (N + 255) / 256;
                nativeVnniGemv_epilogue_m2<<<eblk, 256, 0, stream>>>(
                    d_C, d_C_existing, d_bias, N, beta);
            }
        }

        return cudaGetLastError() == cudaSuccess;
    }

    template <uint8_t CB>
    bool launchKparM2(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        int cuda_device_id, cudaStream_t stream)
    {
        const auto t = selectKparTuning<CB>(N, K);
        const int tw = t.target_waves;
        const int mkg = t.min_kgroups_per_cta;
        const int mkb = t.max_kb;

        switch (t.tile)
        {
        case KparTile::T32_C1:
            return launchKparM2Impl<32, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C1:
            return launchKparM2Impl<64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C2:
            return launchKparM2Impl<64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C1:
            return launchKparM2Impl<128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C2:
            return launchKparM2Impl<128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T256_C2:
            return launchKparM2Impl<256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        }
        return false;
    }

    template <int M, int TILE_N, int CPT, uint8_t CB>
    bool launchKparSmallMImpl(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int target_waves, int min_kgroups_per_cta, int max_kb,
        int force_two_phase,
        int device_id, cudaStream_t stream,
        CUDAGemvContext_ *gemv_ctx)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const int kb = selectKSplit(grid_n, k_groups, num_sms,
                                    target_waves, min_kgroups_per_cta);
        const bool deterministic = llaminar2::debugEnv().gemm.deterministic;
        const int kb_capped_auto = (max_kb > 0) ? std::min(kb, max_kb) : kb;
        const int kb_capped = deterministic ? 1 : kb_capped_auto;

        const size_t partials_bytes = static_cast<size_t>(kb_capped) * M * N * sizeof(float);
        bool use_two_phase;
        if (deterministic)
            use_two_phase = true;
        else if (force_two_phase == 1)
            use_two_phase = true;
        else if (force_two_phase == 2)
            use_two_phase = false;
        else
            use_two_phase = (partials_bytes <= kTwoPhaseMaxBytes);

        dim3 grid(grid_n, kb_capped);

        if (use_two_phase)
        {
            float *d_partials = getKparPartials(
                gemv_ctx, static_cast<size_t>(kb_capped) * M * N);
            if (!d_partials)
                return false;

            nativeVnniGemv_kpar_small_m<M, TILE_N, CPT, CB, true><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_partials, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            const int rblk = (N + 255) / 256;
            nativeVnniGemv_reduce_small_m<M><<<rblk, 256, 0, stream>>>(
                d_partials, d_C, d_C_existing, d_bias, N, kb_capped, beta);
        }
        else
        {
            cudaMemsetAsync(d_C, 0, static_cast<size_t>(M) * N * sizeof(float), stream);

            nativeVnniGemv_kpar_small_m<M, TILE_N, CPT, CB, false><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            if ((beta != 0.0f && d_C_existing) || d_bias)
            {
                const int eblk = (N + 255) / 256;
                nativeVnniGemv_epilogue_small_m<M><<<eblk, 256, 0, stream>>>(
                    d_C, d_C_existing, d_bias, N, beta);
            }
        }

        return cudaGetLastError() == cudaSuccess;
    }

    template <int M, uint8_t CB>
    bool launchKparSmallM(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        int cuda_device_id, cudaStream_t stream)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        const auto t = selectKparTuning<CB>(N, K);
        const int tw = t.target_waves;
        const int mkg = t.min_kgroups_per_cta;
        const int mkb = t.max_kb;

        switch (t.tile)
        {
        case KparTile::T32_C1:
            return launchKparSmallMImpl<M, 32, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C1:
            return launchKparSmallMImpl<M, 64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C2:
            return launchKparSmallMImpl<M, 64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C1:
            return launchKparSmallMImpl<M, 128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C2:
            return launchKparSmallMImpl<M, 128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        case KparTile::T256_C2:
            return launchKparSmallMImpl<M, 256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, cuda_device_id, stream, gemv_ctx);
        }
        return false;
    }

    template <uint8_t CB>
    bool launchKpar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        int cuda_device_id, cudaStream_t stream)
    {
        const auto t = selectKparTuning<CB>(N, K);
        const int tw = t.target_waves;
        const int mkg = t.min_kgroups_per_cta;
        const int mkb = t.max_kb;

        switch (t.tile)
        {
        case KparTile::T32_C1:
            return launchKparImpl<32, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C1:
            return launchKparImpl<64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C2:
            return launchKparImpl<64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C1:
            return launchKparImpl<128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C2:
            return launchKparImpl<128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream, gemv_ctx);
        case KparTile::T256_C2:
            return launchKparImpl<256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream, gemv_ctx);
        }
        return false;
    }

    template <uint8_t CB>
    bool launchDirect(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        cudaStream_t stream)
    {
        const auto t = selectWideTuning(N, K);

        switch (t.tile)
        {
        case WideTile::T32_C1:
        {
            const int grid_n = (N + 32 - 1) / 32;
            nativeVnniGemv_wide<32, 1, CB><<<grid_n, 32, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T64_C1:
        {
            const int grid_n = (N + 64 - 1) / 64;
            nativeVnniGemv_wide<64, 1, CB><<<grid_n, 64, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T64_C2:
        {
            const int grid_n = (N + 64 - 1) / 64;
            nativeVnniGemv_wide<64, 2, CB><<<grid_n, 32, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        case WideTile::T128_C1:
        {
            const int grid_n = (N + 128 - 1) / 128;
            nativeVnniGemv_wide<128, 1, CB><<<grid_n, 128, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C, d_scales_A,
                N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }
        default:
            return false;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    // =====================================================================
    // Row-parallel launch — grid = N blocks, NWARPS warps per block.
    // Requires row-major weight layout (from row-major cache).
    // =====================================================================
    template <uint8_t CB>
    bool launchRowPar(
        const int8_t *d_A_int8,
        const uint8_t *d_payload_rm,
        const uint16_t *d_scales_rm,
        const uint16_t *d_mins_rm,
        const uint32_t *d_emins_rm,
        float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int nwarps,
        cudaStream_t stream)
    {
        switch (nwarps)
        {
        case 2:
            nativeVnniGemv_rowpar<2, CB><<<N, 64, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 4:
            nativeVnniGemv_rowpar<4, CB><<<N, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 8:
            nativeVnniGemv_rowpar<8, CB><<<N, 256, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        default:
            nativeVnniGemv_rowpar<4, CB><<<N, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    template <uint8_t CB>
    bool launchRowParM2(
        const int8_t *d_A_int8,
        const uint8_t *d_payload_rm,
        const uint16_t *d_scales_rm,
        const uint16_t *d_mins_rm,
        const uint32_t *d_emins_rm,
        float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int nwarps,
        cudaStream_t stream)
    {
        switch (nwarps)
        {
        case 2:
            nativeVnniGemv_rowpar_m2<2, CB><<<N, 64, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 4:
            nativeVnniGemv_rowpar_m2<4, CB><<<N, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 8:
            nativeVnniGemv_rowpar_m2<8, CB><<<N, 256, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        default:
            nativeVnniGemv_rowpar_m2<4, CB><<<N, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    template <int M, uint8_t CB>
    bool launchRowParSmallM(
        const int8_t *d_A_int8,
        const uint8_t *d_payload_rm,
        const uint16_t *d_scales_rm,
        const uint16_t *d_mins_rm,
        const uint32_t *d_emins_rm,
        float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int nwarps,
        cudaStream_t stream)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        switch (nwarps)
        {
        case 2:
            nativeVnniGemv_rowpar_small_m<M, 2, CB><<<N, 64, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 4:
            nativeVnniGemv_rowpar_small_m<M, 4, CB><<<N, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 8:
            nativeVnniGemv_rowpar_small_m<M, 8, CB><<<N, 256, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        default:
            nativeVnniGemv_rowpar_small_m<M, 4, CB><<<N, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        }

        return cudaGetLastError() == cudaSuccess;
    }

    // =====================================================================
    template <uint8_t CB>
    bool dispatchGeneratedTuning(
        NativeGemvShape shape,
        const GeneratedDispatchTuning &tuning,
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        CUDARowMajorWeights_ *rowmajor,
        int cuda_device_id, cudaStream_t stream)
    {
        const int tile_key = tuning.tile_n * 100 + tuning.cpt;

        if (shape == NativeGemvShape::WIDE || shape == NativeGemvShape::DIRECT)
        {
            switch (tile_key)
            {
            case 32 * 100 + 1:
            {
                const int grid_n = (N + 32 - 1) / 32;
                nativeVnniGemv_wide<32, 1, CB><<<grid_n, 32, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            case 64 * 100 + 1:
            {
                const int grid_n = (N + 64 - 1) / 64;
                nativeVnniGemv_wide<64, 1, CB><<<grid_n, 64, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            case 64 * 100 + 2:
            {
                const int grid_n = (N + 64 - 1) / 64;
                nativeVnniGemv_wide<64, 2, CB><<<grid_n, 32, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            case 128 * 100 + 1:
            {
                const int grid_n = (N + 128 - 1) / 128;
                nativeVnniGemv_wide<128, 1, CB><<<grid_n, 128, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            case 128 * 100 + 2:
            {
                const int grid_n = (N + 128 - 1) / 128;
                nativeVnniGemv_wide<128, 2, CB><<<grid_n, 64, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            case 256 * 100 + 2:
            {
                const int grid_n = (N + 256 - 1) / 256;
                nativeVnniGemv_wide<256, 2, CB><<<grid_n, 128, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            case 256 * 100 + 4:
            {
                const int grid_n = (N + 256 - 1) / 256;
                nativeVnniGemv_wide<256, 4, CB><<<grid_n, 64, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            case 512 * 100 + 4:
            {
                const int grid_n = (N + 512 - 1) / 512;
                nativeVnniGemv_wide<512, 4, CB><<<grid_n, 128, 0, stream>>>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias);
                return cudaGetLastError() == cudaSuccess;
            }
            default:
                return false;
            }
        }

        if (shape == NativeGemvShape::KPAR)
        {
            switch (tile_key)
            {
            case 32 * 100 + 1:
                return sweepLaunchKpar<32, 1, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 64 * 100 + 1:
                return sweepLaunchKpar<64, 1, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 64 * 100 + 2:
                return sweepLaunchKpar<64, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 128 * 100 + 1:
                return sweepLaunchKpar<128, 1, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 128 * 100 + 2:
                return sweepLaunchKpar<128, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 256 * 100 + 2:
                return sweepLaunchKpar<256, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 256 * 100 + 4:
                return sweepLaunchKpar<256, 4, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            default:
                return false;
            }
        }

        if (shape == NativeGemvShape::ROWPAR)
        {
            if (!rowmajor || !rowmajor->d_payload || !rowmajor->d_scales)
                return false;

            // tile_n encodes NWARPS for ROWPAR (2, 4, 8)
            return launchRowPar<CB>(
                d_A_int8, rowmajor->d_payload, rowmajor->d_scales,
                rowmajor->d_mins, rowmajor->d_emins, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.tile_n, stream);
        }

        return false;
    }

    template <int TILE_N, int CPT, uint8_t CB>
    bool sweepLaunchKpar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int target_waves, int min_kgroups_per_cta, int max_kb,
        int force_two_phase,
        CUDAGemvContext_ *gemv_ctx,
        int device_id, cudaStream_t stream);

    template <uint8_t CB>
    bool dispatchKparM2Generated(
        const GeneratedDispatchTuning &tuning,
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        int cuda_device_id, cudaStream_t stream)
    {
        switch (tuning.tile_n * 100 + tuning.cpt)
        {
        case 32 * 100 + 1:
            return launchKparM2Impl<32, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 1:
            return launchKparM2Impl<64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 2:
            return launchKparM2Impl<64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 1:
            return launchKparM2Impl<128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 2:
            return launchKparM2Impl<128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 256 * 100 + 2:
            return launchKparM2Impl<256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        default:
            return false;
        }
    }

    template <int M, uint8_t CB>
    bool dispatchKparSmallMGenerated(
        const GeneratedDispatchTuning &tuning,
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        int cuda_device_id, cudaStream_t stream)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        switch (tuning.tile_n * 100 + tuning.cpt)
        {
        case 32 * 100 + 1:
            return launchKparSmallMImpl<M, 32, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 1:
            return launchKparSmallMImpl<M, 64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 2:
            return launchKparSmallMImpl<M, 64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 1:
            return launchKparSmallMImpl<M, 128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 2:
            return launchKparSmallMImpl<M, 128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 256 * 100 + 2:
            return launchKparSmallMImpl<M, 256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        default:
            return false;
        }
    }

    // =====================================================================
    // Per-codebook dispatcher — selects kernel family based on shape
    // When g_sweep.active, routes to sweepLaunchKpar / wide with overridden params.
    // =====================================================================
    template <uint8_t CB>
    bool dispatchCodebook(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        CUDARowMajorWeights_ **rm_slot,
        int cuda_device_id, cudaStream_t stream)
    {
        // Sweep override — bypass heuristics, use explicit params
        if (g_sweep.active)
        {
            NativeGemvShape shape = NativeGemvShape::KPAR;
            switch (g_sweep.kernel_family)
            {
            case 0:
                shape = NativeGemvShape::WIDE;
                break;
            case 1:
                shape = NativeGemvShape::KPAR;
                break;
            case 2:
                shape = NativeGemvShape::DIRECT;
                break;
            case 3:
                shape = NativeGemvShape::ROWPAR;
                break;
            default:
                return false;
            }

            const GeneratedDispatchTuning tuning{
                g_sweep.tile_n,
                g_sweep.cpt,
                g_sweep.target_waves,
                g_sweep.mkg,
                g_sweep.max_kb,
                g_sweep.force_two_phase,
            };

            // For ROWPAR sweep: ensure row-major exists (skip for Q8_0
            // which uses column-major ROWPAR to avoid doubling VRAM)
            if (shape == NativeGemvShape::ROWPAR)
            {
                if constexpr (CB != 19)
                {
                    if (rm_slot && !*rm_slot)
                    {
                        constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
                        *rm_slot = cudaRowMajorWeights_create(
                            d_payload, d_scales, d_mins, d_emins,
                            N, K, PB, cuda_device_id, stream);
                    }
                    if (!rm_slot || !*rm_slot)
                        return false;
                }
            }

            recordGemvDispatch<CB>(
                shape,
                tuning,
                N,
                K,
                rm_slot && *rm_slot && (*rm_slot)->d_payload,
                cuda_device_id);

            return dispatchGeneratedTuning<CB>(
                shape, tuning,
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                gemv_ctx, rm_slot ? *rm_slot : nullptr,
                cuda_device_id, stream);
        }

        // Normal dispatch path
        NativeGemvShape shape = classifyShapeGenerated<CB>(N, K);
        GeneratedDispatchTuning tuning = selectGeneratedTuning<CB>(N, K);

        // Row-parallel override: for KPAR shapes, use ROWPAR if available.
        // Q8_0 (CB==19): SKIP — 32-byte payloads cause stride of N*32
        // between adjacent threads in column-major ROWPAR (uncoalesced).
        // KPAR has naturally coalesced column-major access and is faster.
        if (shape == NativeGemvShape::KPAR && isRowParEnabled())
        {
            if constexpr (CB != 19)
            {
                // Lazy-create row-major transpose on first ROWPAR dispatch
                if (rm_slot && !*rm_slot)
                {
                    constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
                    *rm_slot = cudaRowMajorWeights_create(
                        d_payload, d_scales, d_mins, d_emins,
                        N, K, PB, cuda_device_id, stream);
                }
                if (rm_slot && *rm_slot && (*rm_slot)->d_payload)
                {
                    shape = NativeGemvShape::ROWPAR;
                    // NWARPS: 2 for most shapes (well-tested), 4 for large-K shapes
                    // where more warps/block helps hide K-loop memory latency
                    const int k_blocks = K / BLOCK_K;
                    tuning.tile_n = (k_blocks >= 256) ? 4 : 2;
                    tuning.cpt = 0;
                }
            }
        }

        recordGemvDispatch<CB>(
            shape,
            tuning,
            N,
            K,
            rm_slot && *rm_slot && (*rm_slot)->d_payload,
            cuda_device_id);

        return dispatchGeneratedTuning<CB>(
            shape, tuning,
            d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
            d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
            gemv_ctx, rm_slot ? *rm_slot : nullptr,
            cuda_device_id, stream);
    }

    template <uint8_t CB>
    bool dispatchCodebookM2RowPar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        CUDARowMajorWeights_ **rm_slot,
        int cuda_device_id, cudaStream_t stream)
    {
        const NativeGemvShape shape = classifyShapeGenerated<CB>(2, N, K);
        const GeneratedDispatchTuning tuning = selectGeneratedTuning<CB>(2, N, K);
        if (shape == NativeGemvShape::WIDE)
            return false;

        if constexpr (CB != 19)
        {
            if (shape == NativeGemvShape::ROWPAR && isRowParEnabled() && rm_slot)
            {
                if (!*rm_slot)
                {
                    constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
                    *rm_slot = cudaRowMajorWeights_create(
                        d_payload, d_scales, d_mins, d_emins,
                        N, K, PB, cuda_device_id, stream);
                }
                if (*rm_slot && (*rm_slot)->d_payload && (*rm_slot)->d_scales)
                {
                    return launchRowParM2<CB>(
                        d_A_int8, (*rm_slot)->d_payload, (*rm_slot)->d_scales,
                        (*rm_slot)->d_mins, (*rm_slot)->d_emins, d_C,
                        d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                        tuning.tile_n, stream);
                }

                cudaGetLastError(); // Clear row-major allocation/launch errors before the KPAR route.
            }
        }

        if (shape == NativeGemvShape::KPAR)
        {
            return dispatchKparM2Generated<CB>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                gemv_ctx, cuda_device_id, stream);
        }

        return false;
    }

    template <int M, uint8_t CB>
    bool dispatchCodebookSmallMRowPar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        CUDARowMajorWeights_ **rm_slot,
        int cuda_device_id, cudaStream_t stream)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        NativeGemvShape shape = NativeGemvShape::KPAR;
        GeneratedDispatchTuning tuning{};

        if (g_sweep.active)
        {
            /**
             * The generated-dispatch trainer drives this path for verifier
             * buckets M=2..4.  Keep the sweep override local to backend
             * tuning: production dispatch still comes from the generated
             * classifier below, while the perf harness can time real KPAR
             * small-M candidates instead of accidentally timing the current
             * generated route and labelling it as every candidate.
             */
            switch (g_sweep.kernel_family)
            {
            case 0:
                shape = NativeGemvShape::WIDE;
                break;
            case 1:
                shape = NativeGemvShape::KPAR;
                break;
            case 2:
                shape = NativeGemvShape::DIRECT;
                break;
            case 3:
                shape = NativeGemvShape::ROWPAR;
                break;
            default:
                return false;
            }

            tuning = GeneratedDispatchTuning{
                g_sweep.tile_n,
                g_sweep.cpt,
                g_sweep.target_waves,
                g_sweep.mkg,
                g_sweep.max_kb,
                g_sweep.force_two_phase,
            };
        }
        else
        {
            shape = classifyShapeGenerated<CB>(M, N, K);
            tuning = selectGeneratedTuning<CB>(M, N, K);
        }

        if (shape == NativeGemvShape::WIDE)
            return false;

        if constexpr (CB != 19)
        {
            if (shape == NativeGemvShape::ROWPAR && isRowParEnabled() && rm_slot)
            {
                if (!*rm_slot)
                {
                    constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
                    *rm_slot = cudaRowMajorWeights_create(
                        d_payload, d_scales, d_mins, d_emins,
                        N, K, PB, cuda_device_id, stream);
                }
                if (*rm_slot && (*rm_slot)->d_payload && (*rm_slot)->d_scales)
                {
                    return launchRowParSmallM<M, CB>(
                        d_A_int8, (*rm_slot)->d_payload, (*rm_slot)->d_scales,
                        (*rm_slot)->d_mins, (*rm_slot)->d_emins, d_C,
                        d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                        tuning.tile_n, stream);
                }

                cudaGetLastError(); // Clear row-major allocation/launch errors before KPAR fallback.
            }
        }

        if (shape == NativeGemvShape::KPAR)
        {
            return dispatchKparSmallMGenerated<M, CB>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                gemv_ctx, cuda_device_id, stream);
        }

        return false;
    }

    template <int M>
    bool dispatchSmallMByCodebook(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        uint8_t codebook_id,
        int cuda_device_id,
        cudaStream_t cuda_stream,
        CUDAGemvContext_ *gemv_ctx,
        CUDARowMajorWeights_ **rm_slot)
    {
        static_assert(M >= 2 && M <= 4, "CUDA native-VNNI small-M GEMV supports M=2..4");
        switch (codebook_id)
        {
        case 0:
            return dispatchCodebookSmallMRowPar<M, 0>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 4:
            return dispatchCodebookSmallMRowPar<M, 4>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 5:
            return dispatchCodebookSmallMRowPar<M, 5>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 6:
            return dispatchCodebookSmallMRowPar<M, 6>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 7:
            return dispatchCodebookSmallMRowPar<M, 7>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 8:
            return dispatchCodebookSmallMRowPar<M, 8>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 9:
            return dispatchCodebookSmallMRowPar<M, 9>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 10:
            return dispatchCodebookSmallMRowPar<M, 10>(d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 11:
            return dispatchCodebookSmallMRowPar<M, 11>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 12:
            return dispatchCodebookSmallMRowPar<M, 12>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 13:
            return dispatchCodebookSmallMRowPar<M, 13>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 14:
            return dispatchCodebookSmallMRowPar<M, 14>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 15:
            return dispatchCodebookSmallMRowPar<M, 15>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 16:
            return dispatchCodebookSmallMRowPar<M, 16>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 17:
            return dispatchCodebookSmallMRowPar<M, 17>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 19:
            return dispatchCodebookSmallMRowPar<M, 19>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        default:
            return false;
        }
    }

    // =====================================================================
    // Sweep helper — KPAR launch with explicit tuning params
    // =====================================================================
    template <int TILE_N, int CPT, uint8_t CB>
    bool sweepLaunchKpar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int target_waves, int min_kgroups_per_cta, int max_kb,
        int force_two_phase,
        CUDAGemvContext_ *gemv_ctx,
        int device_id, cudaStream_t stream)
    {
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const int kb = selectKSplit(grid_n, k_groups, num_sms,
                                    target_waves, min_kgroups_per_cta);
        const bool deterministic = llaminar2::debugEnv().gemm.deterministic;
        const int kb_capped_auto = (max_kb > 0) ? std::min(kb, max_kb) : kb;
        const int kb_capped = deterministic ? 1 : kb_capped_auto;

        const size_t partials_bytes = static_cast<size_t>(kb_capped) * N * sizeof(float);
        bool use_two_phase;
        if (deterministic)
            use_two_phase = true;
        else if (force_two_phase == 1)
            use_two_phase = true;
        else if (force_two_phase == 2)
            use_two_phase = false;
        else
            use_two_phase = (partials_bytes <= kTwoPhaseMaxBytes);

        dim3 grid(grid_n, kb_capped);

        if (use_two_phase)
        {
            float *d_partials = getKparPartials(
                gemv_ctx, static_cast<size_t>(kb_capped) * N);
            if (!d_partials)
                return false;

            nativeVnniGemv_kpar<TILE_N, CPT, CB, true><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_partials, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            const int rblk = (N + 255) / 256;
            nativeVnniGemv_reduce<<<rblk, 256, 0, stream>>>(
                d_partials, d_C, d_C_existing, d_bias, N, kb_capped, beta);
        }
        else
        {
            cudaMemsetAsync(d_C, 0, static_cast<size_t>(N) * sizeof(float), stream);

            nativeVnniGemv_kpar<TILE_N, CPT, CB, false><<<grid, THREADS, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, kb_capped, alpha);
            {
                cudaError_t le = cudaGetLastError();
                if (le != cudaSuccess)
                    return false;
            }

            if ((beta != 0.0f && d_C_existing) || d_bias)
            {
                const int eblk = (N + 255) / 256;
                nativeVnniGemv_epilogue<<<eblk, 256, 0, stream>>>(
                    d_C, d_C_existing, d_bias, N, beta);
            }
        }

        return cudaGetLastError() == cudaSuccess;
    }
}

// =========================================================================
// Public API — matches original cudaNativeVNNIGemv_fp32 signature
// =========================================================================
extern "C"
{
    bool cudaNativeVNNIGemvTuned_supportsCodebook(uint8_t codebook_id)
    {
        switch (codebook_id)
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
            return true;
        default:
            return false;
        }
    }

    bool cudaNativeVNNIGemvTuned_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot)
    {
        if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
            return false;
        if (N <= 0 || K <= 0 || (K % BLOCK_K) != 0)
            return false;
        if (!cudaNativeVNNIGemvTuned_supportsCodebook(codebook_id))
            return false;
        if (!gemv_ctx)
            return false;

        cudaError_t err = cudaSetDevice(cuda_device_id);
        if (err != cudaSuccess)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        switch (codebook_id)
        {
        case 0:
            return dispatchCodebook<0>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 4:
            return dispatchCodebook<4>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 5:
            return dispatchCodebook<5>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 6:
            return dispatchCodebook<6>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 7:
            return dispatchCodebook<7>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 8:
            return dispatchCodebook<8>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 9:
            return dispatchCodebook<9>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 10:
            return dispatchCodebook<10>(d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 11:
            return dispatchCodebook<11>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 12:
            return dispatchCodebook<12>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 13:
            return dispatchCodebook<13>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 14:
            return dispatchCodebook<14>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 15:
            return dispatchCodebook<15>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 16:
            return dispatchCodebook<16>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 17:
            return dispatchCodebook<17>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 19:
            return dispatchCodebook<19>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        default:
            return false;
        }
    }

    bool cudaNativeVNNIGemvTuned_small_m_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int M,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot)
    {
        if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
            return false;
        if (M < 2 || M > 4 || N <= 0 || K <= 0 || (K % BLOCK_K) != 0)
            return false;
        if (!cudaNativeVNNIGemvTuned_supportsCodebook(codebook_id))
            return false;
        if (!gemv_ctx)
            return false;

        cudaError_t err = cudaSetDevice(cuda_device_id);
        if (err != cudaSuccess)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        switch (M)
        {
        case 2:
            return dispatchSmallMByCodebook<2>(d_A_int8, d_payload, d_scales, d_mins, d_emins,
                                               d_C_fp32, d_scales_A_block, N, K, alpha, beta,
                                               d_C_existing, d_bias, codebook_id, cuda_device_id,
                                               cuda_stream, gemv_ctx, rm_slot);
        case 3:
            return dispatchSmallMByCodebook<3>(d_A_int8, d_payload, d_scales, d_mins, d_emins,
                                               d_C_fp32, d_scales_A_block, N, K, alpha, beta,
                                               d_C_existing, d_bias, codebook_id, cuda_device_id,
                                               cuda_stream, gemv_ctx, rm_slot);
        case 4:
            return dispatchSmallMByCodebook<4>(d_A_int8, d_payload, d_scales, d_mins, d_emins,
                                               d_C_fp32, d_scales_A_block, N, K, alpha, beta,
                                               d_C_existing, d_bias, codebook_id, cuda_device_id,
                                               cuda_stream, gemv_ctx, rm_slot);
        default:
            return false;
        }
    }

    bool cudaNativeVNNIGemvTuned_m2_fp32(
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C_fp32,
        const float *d_scales_A_block,
        int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        uint8_t codebook_id,
        int cuda_device_id,
        void *stream,
        CUDAGemvContext *gemv_ctx,
        CUDARowMajorWeights **rm_slot)
    {
        return cudaNativeVNNIGemvTuned_small_m_fp32(
            d_A_int8,
            d_payload,
            d_scales,
            d_mins,
            d_emins,
            d_C_fp32,
            d_scales_A_block,
            2,
            N,
            K,
            alpha,
            beta,
            d_C_existing,
            d_bias,
            codebook_id,
            cuda_device_id,
            stream,
            gemv_ctx,
            rm_slot);

    }

    bool cudaNativeVNNIInitIQGridTables_tuned()
    {
        return llaminar2::cuda_native_vnni::initIQGridTables();
    }

    // =====================================================================
    // Sweep config API — set/clear the global override that dispatchCodebook
    // reads to bypass heuristics during automated tuning sweeps.
    //
    // kernel_family: 0=WIDE, 1=KPAR, 2=DIRECT
    // force_two_phase: 0=auto, 1=force two-phase, 2=force atomic
    // =====================================================================
    void cudaNativeVNNIGemvSweep_setConfig(
        int kernel_family, int tile_n, int cpt,
        int target_waves, int mkg, int max_kb,
        int force_two_phase)
    {
        g_sweep.active = true;
        g_sweep.kernel_family = kernel_family;
        g_sweep.tile_n = tile_n;
        g_sweep.cpt = cpt;
        g_sweep.target_waves = target_waves;
        g_sweep.mkg = mkg;
        g_sweep.max_kb = max_kb;
        g_sweep.force_two_phase = force_two_phase;
    }

    void cudaNativeVNNIGemvSweep_clearConfig()
    {
        g_sweep.active = false;
    }

    bool cudaNativeVNNIGemvSweep_isActive()
    {
        return g_sweep.active;
    }

    void cudaNativeVNNIGemvTuned_clearStaticState()
    {
        // Row-major weight data is now owned by CUDAPackedWeights (per-weight)
        // and KPAR partials are owned by CUDAGemvContext (per-device).
        // Only sweep override needs clearing here.
        g_sweep.active = false;
    }

    // =================================================================
    // Context lifecycle — extern "C" create/destroy
    // =================================================================

    CUDAGemvContext *cudaGemvContext_create(int device_id)
    {
        auto *ctx = new (std::nothrow) CUDAGemvContext_();
        if (!ctx)
            return nullptr;
        ctx->device_id = device_id;
        return ctx;
    }

    void cudaGemvContext_destroy(CUDAGemvContext *ctx)
    {
        if (!ctx)
            return;
        delete ctx;
    }

    void cudaGemvContext_bindWorkspace(
        CUDAGemvContext *ctx,
        float *kpar_partials,
        size_t kpar_partials_bytes)
    {
        if (!ctx)
            return;
        ctx->kpar_partials = kpar_partials;
        ctx->kpar_capacity = kpar_partials_bytes / sizeof(float);
    }

    CUDARowMajorWeights *cudaRowMajorWeights_create(
        const uint8_t *d_payload_col,
        const uint16_t *d_scales_col,
        const uint16_t *d_mins_col,
        const uint32_t *d_emins_col,
        int N, int K,
        int payload_bytes,
        int device_id,
        void *stream)
    {
        cudaSetDevice(device_id);
        cudaStream_t s = static_cast<cudaStream_t>(stream);
        const int K_blocks = K / BLOCK_K;

        auto *rm = new (std::nothrow) CUDARowMajorWeights_();
        if (!rm)
            return nullptr;
        rm->N = N;
        rm->K_blocks = K_blocks;
        rm->device_id = device_id;

        // Transpose payload (PB bytes per block)
        switch (payload_bytes)
        {
        case 2:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 2>(d_payload_col, N, K_blocks, s));
            break;
        case 4:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 4>(d_payload_col, N, K_blocks, s));
            break;
        case 16:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 16>(d_payload_col, N, K_blocks, s));
            break;
        case 32:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 32>(d_payload_col, N, K_blocks, s));
            break;
        default:
            delete rm;
            return nullptr;
        }
        if (!rm->d_payload)
        {
            delete rm;
            return nullptr;
        }

        // Transpose scales (2 bytes each)
        rm->d_scales = transposeBuffer<uint16_t, 2>(d_scales_col, N, K_blocks, s);
        if (!rm->d_scales)
        {
            cudaFree(rm->d_payload);
            delete rm;
            return nullptr;
        }

        // Transpose mins if present
        if (d_mins_col)
        {
            rm->d_mins = transposeBuffer<uint16_t, 2>(d_mins_col, N, K_blocks, s);
            if (!rm->d_mins)
            {
                cudaFree(rm->d_payload);
                cudaFree(rm->d_scales);
                delete rm;
                return nullptr;
            }
        }

        // Transpose emins if present
        if (d_emins_col)
        {
            rm->d_emins = transposeBuffer<uint32_t, 4>(d_emins_col, N, K_blocks, s);
            if (!rm->d_emins)
            {
                cudaFree(rm->d_payload);
                cudaFree(rm->d_scales);
                if (rm->d_mins)
                    cudaFree(rm->d_mins);
                delete rm;
                return nullptr;
            }
        }

        cudaStreamSynchronize(s);
        return rm;
    }

    void cudaRowMajorWeights_destroy(CUDARowMajorWeights *rm)
    {
        if (!rm)
            return;
        cudaSetDevice(rm->device_id);
        if (rm->d_payload)
            cudaFree(rm->d_payload);
        if (rm->d_scales)
            cudaFree(rm->d_scales);
        if (rm->d_mins)
            cudaFree(rm->d_mins);
        if (rm->d_emins)
            cudaFree(rm->d_emins);
        delete rm;
    }
}
