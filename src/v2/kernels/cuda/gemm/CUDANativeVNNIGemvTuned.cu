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
 *   Small non-verifier partial sets (≤256KB) use two phases: each CTA writes
 *     `d_partials[split_idx * N + n]`, then an ordered reducer publishes `d_C[n]`.
 *   Large non-verifier partial sets may use atomic accumulation when the caller
 *     does not require batch invariance. Publication-sensitive serial M=1 and
 *     grouped verifier work always materializes partitions and reduces
 *     them in increasing partition order. The verifier contract therefore keeps
 *     K parallelism without allowing scheduler-dependent floating-point atomics.
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
#include "utils/PrefillGraphBucketDefaults.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <mutex>

static thread_local int g_cuda_native_vnni_decode_equivalent_m1_config = 0;

static bool decodeEquivalentM1ConfigActive()
{
    return g_cuda_native_vnni_decode_equivalent_m1_config != 0;
}

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
// for coalesced K-dimension reads. This representation is created only by an
// explicitly forced diagnostic ROWPAR candidate and stored on CUDAPackedWeights
// until the weight tensor is destroyed. Production dispatch must not allocate
// it or override a generated policy decision implicitly.
//
// Memory cost: ~1× the weight data (about 1.4 GB for 7B Q4_0).
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
        int exact_kb = 0;
        int force_two_phase = 0; // 0=auto, 1=force 2-phase, 2=force atomic
    };
    static SweepOverride g_sweep;
    static thread_local int g_last_effective_kb = 1;

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

    /**
     * @brief Accumulate one 32-element native-VNNI weight block exactly once.
     *
     * MTP verifier publication compares grouped runtime-M rows against serial
     * M=1 decode route bit-for-bit. CUDA may otherwise choose subtly different
     * multiply/add contraction when the same expression lives in the larger
     * grouped kernel body. This helper gives both the M=1 KPAR kernel and the
     * grouped small-M KPAR kernel one shared, explicit round-to-nearest sequence
     * for the per-block FP32 contribution.
     */
    template <uint8_t CB>
    __device__ __forceinline__ float native_vnni_block_contribution_rn(
        const int32_t *__restrict__ a_vals,
        const int32_t *__restrict__ packed_groups,
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        size_t linear,
        float scale_a)
    {
        if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
        {
            int dot_lo = 0;
            int dot_hi = 0;
            int sum_lo = 0;
            int sum_hi = 0;
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
            const float dot_term = __fadd_rn(
                __fmul_rn(scale_lo, static_cast<float>(dot_lo)),
                __fmul_rn(scale_hi, static_cast<float>(dot_hi)));
            float contribution = __fmul_rn(scale_a, dot_term);

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
            {
                const uint32_t emin = d_emins ? d_emins[linear] : 0u;
                const float min_lo = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin));
                const float min_hi = llaminar2::cuda_native_vnni::fp16_bits_to_float(static_cast<uint16_t>(emin >> 16));
                const float min_term = __fadd_rn(
                    __fmul_rn(min_lo, static_cast<float>(sum_lo)),
                    __fmul_rn(min_hi, static_cast<float>(sum_hi)));
                contribution = __fadd_rn(contribution, __fmul_rn(scale_a, min_term));
            }

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_iq1_m)
            {
                constexpr float IQ1S_DELTA = 0.125f;
                const uint8_t qh0 = payload[4];
                const uint8_t qh1 = payload[5];
                const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[0]) +
                                llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[1]);
                const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[2]) +
                                llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[3]);
                const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[4]) +
                                llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[5]);
                const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[6]) +
                                llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[7]);
                const float d0 = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                const float d1 = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                const float d2 = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA;
                const float d3 = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA;
                const float lo_delta = __fmul_rn(
                    __fadd_rn(__fmul_rn(d0, static_cast<float>(sg0)),
                              __fmul_rn(d1, static_cast<float>(sg1))),
                    scale_lo);
                const float hi_delta = __fmul_rn(
                    __fadd_rn(__fmul_rn(d2, static_cast<float>(sg2)),
                              __fmul_rn(d3, static_cast<float>(sg3))),
                    scale_hi);
                contribution = __fadd_rn(
                    contribution,
                    __fmul_rn(scale_a, __fadd_rn(lo_delta, hi_delta)));
            }

            return contribution;
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
            float contribution = __fmul_rn(
                __fmul_rn(scale_a, scale_b),
                static_cast<float>(dot));

            if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
            {
                const float min_b = d_mins ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins[linear]) : 0.0f;
                contribution = __fadd_rn(
                    contribution,
                    __fmul_rn(__fmul_rn(scale_a, min_b), static_cast<float>(sum_a)));
            }

            return contribution;
        }
    }

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
            cudaGetLastError(); // Clear sticky state before reporting preparation failure.
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

    static int resolveKparKBlocks(
        int grid_n,
        int k_groups,
        int num_sms,
        int target_waves,
        int min_kgroups_per_cta,
        int max_kb,
        int exact_kb);

    /**
     * @brief Resolve the execution surface represented by the active stream.
     * @param stream CUDA stream on which the NativeVNNI launch will be issued.
     * @param graph_captured Receives true while the stream is being captured.
     * @return true when CUDA reported a trustworthy capture state.
     *
     * Policy ABI v2 treats eager launch gaps and graph-captured replay as
     * distinct economical surfaces. A failed runtime query therefore cannot be
     * interpreted as eager execution: doing so could select a different exact
     * K partition and break the verifier's serial-M1 arithmetic identity.
     */
    static bool queryGraphCapturedExecution(
        cudaStream_t stream,
        bool &graph_captured)
    {
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        const cudaError_t error = cudaStreamIsCapturing(stream, &status);
        if (error != cudaSuccess)
            return false;
        graph_captured = status != cudaStreamCaptureStatusNone;
        return true;
    }

    /**
     * @brief Build the stable common-trainer identity for an observed launch.
     *
     * KPAR identity includes the effective partition count because changing KB
     * changes FP32 parenthesization. Grouped verifier rows remain one semantic
     * candidate: they inherit the complete frozen public-M1 schedule, including
     * family, tile, and KB. Keeping this identity in production telemetry stops
     * a trainer from treating target-wave spellings that resolve to the same
     * launch as independent evidence.
     */
    static std::string effectiveCandidateName(
        NativeGemvShape shape,
        const GeneratedDispatchTuning &tuning,
        int effective_kb,
        int M,
        bool verifier_serial_m1)
    {
        if (verifier_serial_m1 && M >= 2)
            return "cuda.nvnni.decode.verifier.inherit_serial_m1";

        if (M == 1 &&
            (shape == NativeGemvShape::WIDE ||
             shape == NativeGemvShape::DIRECT ||
             shape == NativeGemvShape::KPAR))
        {
            std::string candidate =
                "cuda.nvnni.decode.fast_m1." +
                std::string(shapeName(shape)) +
                ".tn" + std::to_string(tuning.tile_n) +
                ".cpt" + std::to_string(tuning.cpt);
            if (shape == NativeGemvShape::KPAR)
                candidate += ".kb" + std::to_string(effective_kb);
            return candidate;
        }

        return "unsupported:" + std::string(shapeName(shape));
    }

    template <uint8_t CB>
    static void recordGemvDispatch(
        NativeGemvShape shape,
        const GeneratedDispatchTuning &tuning,
        int M,
        int N,
        int K,
        bool rowmajor_available,
        int cuda_device_id,
        bool graph_captured,
        bool verifier_serial_m1,
        CUDAGemvContext_ *gemv_ctx)
    {
        if (!llaminar2::PerfStatsCollector::isEnabled())
            return;

        int effective_kb = 1;
        if (shape == NativeGemvShape::KPAR)
        {
            effective_kb = resolveKparKBlocks(
                (N + tuning.tile_n - 1) / tuning.tile_n,
                K / BLOCK_K,
                querySmCount(gemv_ctx),
                tuning.target_waves,
                tuning.mkg,
                tuning.max_kb,
                tuning.exact_kb);
        }

        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "cuda_native_vnni_gemv_dispatch",
            1.0,
            "decode",
            "cuda:" + std::to_string(cuda_device_id),
            llaminar2::PerfStatsCollector::Tags{
                {"codebook", std::to_string(static_cast<int>(CB))},
                {"m", std::to_string(M)},
                {"n", std::to_string(N)},
                {"k", std::to_string(K)},
                {"execution_mode", graph_captured
                                       ? "graph_captured"
                                       : "eager"},
                {"semantic_contract", verifier_serial_m1
                                             ? "verifier_serial_m1_bitwise"
                                             : "fast"},
                {"effective_candidate_id",
                 effectiveCandidateName(
                     shape, tuning, effective_kb, M, verifier_serial_m1)},
                {"route", shapeName(shape)},
                {"tile_n", std::to_string(tuning.tile_n)},
                {"cpt", std::to_string(tuning.cpt)},
                {"effective_kb", std::to_string(effective_kb)},
                {"target_waves", std::to_string(tuning.target_waves)},
                {"mkg", std::to_string(tuning.mkg)},
                {"max_kb", std::to_string(tuning.max_kb)},
                {"exact_kb", std::to_string(tuning.exact_kb)},
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

    /**
     * @brief Resolve the effective K-partition count for one launch.
     *
     * Generated production policies carry the exact KB selected by the common
     * trainer so every installed candidate has one unambiguous reduction tree.
     * Legacy generated rows may leave exact KB at zero, in which case the old
     * occupancy target and cap remain sufficient to read the transition table.
     * Values larger than the number of 32-element K groups are rejected rather
     * than silently clamped and mislabeled.
     */
    static int resolveKparKBlocks(
        int grid_n,
        int k_groups,
        int num_sms,
        int target_waves,
        int min_kgroups_per_cta,
        int max_kb,
        int exact_kb)
    {
        if (grid_n <= 0 || k_groups <= 0 || num_sms <= 0)
            return 0;

        int kb = 0;
        if (exact_kb > 0)
        {
            if (exact_kb > k_groups)
                return 0;
            kb = exact_kb;
        }
        else
        {
            kb = selectKSplit(
                grid_n, k_groups, num_sms,
                target_waves, min_kgroups_per_cta);
            if (max_kb > 0)
                kb = std::min(kb, max_kb);
        }

        if (llaminar2::debugEnv().gemm.deterministic)
            return 1;
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

                const float contribution = native_vnni_block_contribution_rn<CB>(
                    smem_A,
                    packed_groups,
                    payload,
                    d_scales,
                    d_mins,
                    d_emins,
                    linear,
                    scale_a);
                acc[c] = __fadd_rn(acc[c], contribution);
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

    /**
     * @brief Grouped verifier equivalent of the serial M=1 WIDE/DIRECT kernel.
     *
     * The grid's Y dimension is the verifier row.  That keeps all rows inside one
     * grouped launch while preserving the row-local arithmetic shape of ordinary
     * serial decode: one CTA tile loads exactly one row's 32-element activation
     * block into shared memory, decodes the same column-major payload, and writes
     * to `[row, n]`.
     */
    template <int TILE_N, int CPT, uint8_t CB>
    __global__ void nativeVnniGemv_wide_small_m_serial_rows(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int M, int N, int K,
        float alpha, float beta,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias)
    {
        const int row = blockIdx.y;
        if (row >= M)
            return;

        const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
        if (n_base >= N)
            return;

        const int blocks_per_row = K / BLOCK_K;
        __shared__ int32_t smem_A[8];

        float acc[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[c] = 0.0f;

        for (int blk = 0; blk < blocks_per_row; ++blk)
        {
            if (threadIdx.x < 8)
            {
                smem_A[threadIdx.x] = *reinterpret_cast<const int32_t *>(
                    d_A_int8 + static_cast<size_t>(row) * K + blk * BLOCK_K + threadIdx.x * 4);
            }
            __syncthreads();

            const float scale_a = d_scales_A[static_cast<size_t>(row) * blocks_per_row + blk];

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

                const float contribution = native_vnni_block_contribution_rn<CB>(
                    smem_A,
                    packed_groups,
                    payload,
                    d_scales,
                    d_mins,
                    d_emins,
                    linear,
                    scale_a);
                acc[c] = __fadd_rn(acc[c], contribution);
            }

            __syncthreads();
        }

#pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                const size_t out_idx = static_cast<size_t>(row) * N + n;
                float out = __fmul_rn(alpha, acc[c]);
                if (beta != 0.0f && d_C_existing)
                    out = __fadd_rn(out, __fmul_rn(beta, d_C_existing[out_idx]));
                if (d_bias)
                    out = __fadd_rn(out, d_bias[n]);
                d_C[out_idx] = out;
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

        float acc[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[c] = 0.0f;

        /*
         * Some exact KB values do not divide the K-block count and therefore
         * create trailing empty partitions. Those partitions must still write
         * explicit zero partials: returning here would leave reusable graph
         * workspace stale and make the ordered reducer nondeterministic.
         */

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

                const float contribution = native_vnni_block_contribution_rn<CB>(
                    a_vals,
                    packed_groups,
                    payload,
                    d_scales,
                    d_mins,
                    d_emins,
                    linear,
                    scale_a);
                acc[c] = __fadd_rn(acc[c], contribution);
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

    /**
     * @brief Runtime-row ROWPAR verifier kernel with pairwise weight reuse.
     *
     * Grid Y selects a fixed-size row tile. Every CTA owns one output column
     * and performs the serial ROWPAR reduction order for each active row while
     * reusing each decoded weight block across the rows in the tile. Runtime M
     * only changes the number of independent row tiles in the grid.
     */
    template <int ROWS_PER_TILE, int NWARPS, uint8_t CB>
    __global__ void nativeVnniGemv_rowpar_small_m(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload_rm,
        const uint16_t *__restrict__ d_scales_rm,
        const uint16_t *__restrict__ d_mins_rm,
        const uint32_t *__restrict__ d_emins_rm,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int M, int N, int K,
        float alpha, float beta,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias)
    {
        static_assert(ROWS_PER_TILE > 0, "Verifier row tiles must contain at least one row");
        constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
        const int n = blockIdx.x;
        const int row_base = blockIdx.y * ROWS_PER_TILE;
        if (n >= N)
            return;

        const int k_blocks = K / BLOCK_K;
        const int tid = threadIdx.x;
        const int lane_id = tid & 31;
        const int warp_id = tid >> 5;

        float acc[ROWS_PER_TILE];
#pragma unroll
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            acc[tile_row] = 0.0f;

        for (int blk = tid; blk < k_blocks; blk += NWARPS * 32)
        {
            int32_t a_vals[ROWS_PER_TILE][8];
#pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const int row = row_base + tile_row;
                if (row >= M)
                    continue;
                const int4 *a_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + static_cast<size_t>(row) * K + blk * BLOCK_K);
                const int4 a_lo = a_ptr128[0];
                const int4 a_hi = a_ptr128[1];
                a_vals[tile_row][0] = a_lo.x;
                a_vals[tile_row][1] = a_lo.y;
                a_vals[tile_row][2] = a_lo.z;
                a_vals[tile_row][3] = a_lo.w;
                a_vals[tile_row][4] = a_hi.x;
                a_vals[tile_row][5] = a_hi.y;
                a_vals[tile_row][6] = a_hi.z;
                a_vals[tile_row][7] = a_hi.w;
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
                for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
                {
                    const int row = row_base + tile_row;
                    if (row >= M)
                        continue;
                    int dot_lo = 0;
                    int dot_hi = 0;
                    int sum_lo = 0;
                    int sum_hi = 0;
#pragma unroll
                    for (int g = 0; g < 4; ++g)
                    {
                        dot_lo = __dp4a(a_vals[tile_row][g], packed_groups[g], dot_lo);
                        dot_hi = __dp4a(a_vals[tile_row][g + 4], packed_groups[g + 4], dot_hi);
                        sum_lo += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][g]);
                        sum_hi += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][g + 4]);
                    }

                    const float scale_a = d_scales_A[static_cast<size_t>(row) * k_blocks + blk];
                    acc[tile_row] += scale_a * (scale_lo * static_cast<float>(dot_lo) +
                                                scale_hi * static_cast<float>(dot_hi));

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale_asym)
                    {
                        acc[tile_row] += scale_a * (min_lo * static_cast<float>(sum_lo) +
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
                        const int sg0 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][0]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][1]);
                        const int sg1 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][2]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][3]);
                        const int sg2 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][4]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][5]);
                        const int sg3 = llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][6]) +
                                        llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][7]);
                        acc[tile_row] += scale_a * ((d0 * static_cast<float>(sg0) + d1 * static_cast<float>(sg1)) * scale_lo +
                                                    (d2 * static_cast<float>(sg2) + d3 * static_cast<float>(sg3)) * scale_hi);
                    }
                }
            }
            else
            {
                const float scale_b = llaminar2::cuda_native_vnni::fp16_bits_to_float(d_scales_rm[linear]);
                const float min_b = d_mins_rm ? llaminar2::cuda_native_vnni::fp16_bits_to_float(d_mins_rm[linear]) : 0.0f;

#pragma unroll
                for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
                {
                    const int row = row_base + tile_row;
                    if (row >= M)
                        continue;
                    int dot = 0;
                    int sum_a = 0;
#pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        dot = __dp4a(a_vals[tile_row][g], packed_groups[g], dot);
                        sum_a += llaminar2::cuda_native_vnni::sum_packed_i8(a_vals[tile_row][g]);
                    }

                    const float scale_a = d_scales_A[static_cast<size_t>(row) * k_blocks + blk];
                    acc[tile_row] += scale_a * scale_b * static_cast<float>(dot);

                    if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_asymmetric)
                    {
                        acc[tile_row] += scale_a * min_b * static_cast<float>(sum_a);
                    }
                }
            }
        }

#pragma unroll
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
        {
            for (int mask = 16; mask > 0; mask >>= 1)
                acc[tile_row] += __shfl_xor_sync(0xFFFFFFFF, acc[tile_row], mask);
        }

        __shared__ float reduce[ROWS_PER_TILE][NWARPS];
        if (lane_id == 0)
        {
#pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
                reduce[tile_row][warp_id] = acc[tile_row];
        }
        __syncthreads();

        if (tid == 0)
        {
#pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const int row = row_base + tile_row;
                if (row >= M)
                    continue;
                float sum = reduce[tile_row][0];
#pragma unroll
                for (int w = 1; w < NWARPS; ++w)
                    sum += reduce[tile_row][w];

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

    /**
     * @brief Runtime-row K-parallel verifier kernel with bounded register use.
     *
     * Each CTA owns one K partition, one output-column tile, and a fixed-size
     * row tile. The fixed row tile lets the CTA decode a weight block once and
     * reuse it across verifier rows without making register pressure grow with
     * the requested speculative depth. Increasing runtime @p M therefore adds
     * independent Z-grid tiles instead of compiling another kernel or changing
     * the arithmetic performed for an existing row.
     */
    template <int ROWS_PER_TILE, int TILE_N, int CPT, uint8_t CB, bool TWO_PHASE = false>
    __global__ void nativeVnniGemv_kpar_small_m(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        float *__restrict__ d_C,
        const float *__restrict__ d_scales_A,
        int M, int N, int K, int kb,
        float alpha)
    {
        static_assert(ROWS_PER_TILE > 0, "Verifier row tiles must contain at least one row");
        const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
        const int split_idx = blockIdx.y;
        const int row_base = blockIdx.z * ROWS_PER_TILE;
        if (n_base >= N)
            return;

        const int blocks_per_row = K / BLOCK_K;
        const int blocks_per_split = (blocks_per_row + kb - 1) / kb;
        const int blk_begin = split_idx * blocks_per_split;
        const int blk_end = min(blocks_per_row, blk_begin + blocks_per_split);

        float acc[ROWS_PER_TILE][CPT];
#pragma unroll
        for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
        {
#pragma unroll
            for (int c = 0; c < CPT; ++c)
                acc[tile_row][c] = 0.0f;
        }

        /*
         * Keep trailing empty K partitions in the publication tree by writing
         * zero for every owned verifier row/column. This is required because
         * the partials arena survives graph replay and may contain an older
         * launch's bytes when the current exact KB has empty tail partitions.
         */

        for (int blk = blk_begin; blk < blk_end; ++blk)
        {
            int32_t a_vals[ROWS_PER_TILE][8];
#pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const int row = row_base + tile_row;
                if (row >= M)
                    continue;
                const int4 *a_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + static_cast<size_t>(row) * K + blk * BLOCK_K);
                const int4 a_lo = a_ptr128[0];
                const int4 a_hi = a_ptr128[1];
                a_vals[tile_row][0] = a_lo.x;
                a_vals[tile_row][1] = a_lo.y;
                a_vals[tile_row][2] = a_lo.z;
                a_vals[tile_row][3] = a_lo.w;
                a_vals[tile_row][4] = a_hi.x;
                a_vals[tile_row][5] = a_hi.y;
                a_vals[tile_row][6] = a_hi.z;
                a_vals[tile_row][7] = a_hi.w;
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

#pragma unroll
                for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
                {
                    const int row = row_base + tile_row;
                    if (row >= M)
                        continue;
                    const float scale_a = d_scales_A[static_cast<size_t>(row) * blocks_per_row + blk];
                    const float contribution = native_vnni_block_contribution_rn<CB>(
                        a_vals[tile_row],
                        packed_groups,
                        payload,
                        d_scales,
                        d_mins,
                        d_emins,
                        linear,
                        scale_a);
                    acc[tile_row][c] = __fadd_rn(acc[tile_row][c], contribution);
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
                for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
                {
                    const int row = row_base + tile_row;
                    if (row >= M)
                        continue;
                    if constexpr (TWO_PHASE)
                    {
                        d_C[(static_cast<size_t>(split_idx) * M + row) * N + n] =
                            alpha * acc[tile_row][c];
                    }
                    else
                    {
                        atomicAdd(&d_C[static_cast<size_t>(row) * N + n],
                                  alpha * acc[tile_row][c]);
                    }
                }
            }
        }
    }

    /**
     * @brief Publish one runtime verifier row from ordered K-partition partials.
     *
     * Grid Y owns the row, so reduction order and register state are independent
     * of the total verifier depth. This is the same ascending-partition loop as
     * serial M=1 publication, applied to the row's disjoint partial slice.
     */
    __global__ void nativeVnniGemv_reduce_small_m(
        const float *__restrict__ d_partials,
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int M, int N, int kb, float beta)
    {
        const int row = blockIdx.y;
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M || n >= N)
            return;

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
        int exact_kb,
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
        int exact_kb,
        int force_two_phase,
        int device_id, cudaStream_t stream,
        CUDAGemvContext_ *gemv_ctx)
    {
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const bool debug_deterministic = llaminar2::debugEnv().gemm.deterministic;
        const bool verifier_decode_equivalent = decodeEquivalentM1ConfigActive();
        const int kb_capped = resolveKparKBlocks(
            grid_n, k_groups, num_sms,
            target_waves, min_kgroups_per_cta, max_kb, exact_kb);
        if (kb_capped <= 0)
            return false;
        /*
         * Global deterministic debugging retains its historical one-partition
         * contract. MTP decode equivalence is different: serial M=1 and grouped
         * grouped verifier rows execute the same generated K-partition geometry, write disjoint
         * partials, and use the same ascending reducer. Collapsing that contract
         * to KB=1 discarded useful output parallelism without improving the
         * arithmetic identity shared by the two production entry points.
         */
        g_last_effective_kb = kb_capped;

        // Choose two-phase vs atomic based on partials buffer size.
        // Verifier publication never permits scheduler-ordered atomic additions.
        const size_t partials_bytes = static_cast<size_t>(kb_capped) * N * sizeof(float);
        const bool use_two_phase =
            debug_deterministic || verifier_decode_equivalent ||
            (partials_bytes <= kTwoPhaseMaxBytes);

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
    bool launchKparSmallMImpl(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int target_waves, int min_kgroups_per_cta, int max_kb,
        int exact_kb,
        int force_two_phase,
        int device_id, cudaStream_t stream,
        CUDAGemvContext_ *gemv_ctx)
    {
        constexpr int ROWS_PER_TILE = 2;
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const int kb_capped = resolveKparKBlocks(
            grid_n, k_groups, num_sms,
            target_waves, min_kgroups_per_cta, max_kb, exact_kb);
        if (kb_capped <= 0)
            return false;
        g_last_effective_kb = kb_capped;

        /*
         * Grouped verifier publication is always two-phase. A force-atomic
         * sweep candidate is rejected instead of silently measuring a schedule-
         * dependent kernel that can never satisfy batch invariance.
         */
        if (force_two_phase == 2 && g_sweep.active)
            return false;

        /*
         * The reduction arena is a reusable row tile, not a semantic M limit.
         * A configured verifier wider than the default graph capacity records
         * several grouped kernel/reducer pairs in the same graph. Each pair
         * still owns multiple rows and reuses one packed-weight traversal; no
         * row is replayed through the scalar public API.
         */
        const int workspace_tile_rows = std::min(
            M,
            llaminar2::kDefaultNativeVNNIVerifierRowCapacity);
        float *d_partials = getKparPartials(
            gemv_ctx,
            static_cast<size_t>(kb_capped) * workspace_tile_rows * N);
        if (!d_partials)
            return false;

        const int activation_blocks_per_row = K / BLOCK_K;
        for (int row_base = 0; row_base < M; row_base += workspace_tile_rows)
        {
            const int tile_m = std::min(workspace_tile_rows, M - row_base);
            const int row_tiles =
                (tile_m + ROWS_PER_TILE - 1) / ROWS_PER_TILE;
            const dim3 grid(grid_n, kb_capped, row_tiles);

            const int8_t *tile_a =
                d_A_int8 + static_cast<size_t>(row_base) * K;
            const float *tile_scales =
                d_scales_A +
                static_cast<size_t>(row_base) * activation_blocks_per_row;
            float *tile_c =
                d_C + static_cast<size_t>(row_base) * N;
            const float *tile_existing =
                d_C_existing
                    ? d_C_existing + static_cast<size_t>(row_base) * N
                    : nullptr;

            nativeVnniGemv_kpar_small_m<ROWS_PER_TILE, TILE_N, CPT, CB, true>
                <<<grid, THREADS, 0, stream>>>(
                    tile_a, d_payload, d_scales, d_mins, d_emins,
                    d_partials, tile_scales, tile_m, N, K, kb_capped, alpha);
            const cudaError_t launch_error = cudaGetLastError();
            if (launch_error != cudaSuccess)
                return false;

            const int reduction_blocks = (N + 255) / 256;
            nativeVnniGemv_reduce_small_m<<<
                dim3(reduction_blocks, tile_m), 256, 0, stream>>>(
                d_partials,
                tile_c,
                tile_existing,
                d_bias,
                tile_m,
                N,
                kb_capped,
                beta);
            if (cudaGetLastError() != cudaSuccess)
                return false;
        }

        return true;
    }

    template <uint8_t CB>
    bool launchKparSmallM(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int M, int N, int K,
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
            return launchKparSmallMImpl<32, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C1:
            return launchKparSmallMImpl<64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C2:
            return launchKparSmallMImpl<64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C1:
            return launchKparSmallMImpl<128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C2:
            return launchKparSmallMImpl<128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T256_C2:
            return launchKparSmallMImpl<256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
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
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C1:
            return launchKparImpl<64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T64_C2:
            return launchKparImpl<64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C1:
            return launchKparImpl<128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T128_C2:
            return launchKparImpl<128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
        case KparTile::T256_C2:
            return launchKparImpl<256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, 0, 0,
                cuda_device_id, stream, gemv_ctx);
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
    bool launchRowParSmallM(
        const int8_t *d_A_int8,
        const uint8_t *d_payload_rm,
        const uint16_t *d_scales_rm,
        const uint16_t *d_mins_rm,
        const uint32_t *d_emins_rm,
        float *d_C,
        const float *d_scales_A, int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        int nwarps,
        cudaStream_t stream)
    {
        constexpr int ROWS_PER_TILE = 2;
        const dim3 grid(N, (M + ROWS_PER_TILE - 1) / ROWS_PER_TILE);
        switch (nwarps)
        {
        case 2:
            nativeVnniGemv_rowpar_small_m<ROWS_PER_TILE, 2, CB><<<grid, 64, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 4:
            nativeVnniGemv_rowpar_small_m<ROWS_PER_TILE, 4, CB><<<grid, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        case 8:
            nativeVnniGemv_rowpar_small_m<ROWS_PER_TILE, 8, CB><<<grid, 256, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            break;
        default:
            nativeVnniGemv_rowpar_small_m<ROWS_PER_TILE, 4, CB><<<grid, 128, 0, stream>>>(
                d_A_int8, d_payload_rm, d_scales_rm, d_mins_rm, d_emins_rm, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
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
        g_last_effective_kb = 1;
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
                    tuning.exact_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 64 * 100 + 1:
                return sweepLaunchKpar<64, 1, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.exact_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 64 * 100 + 2:
                return sweepLaunchKpar<64, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.exact_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 128 * 100 + 1:
                return sweepLaunchKpar<128, 1, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.exact_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 128 * 100 + 2:
                return sweepLaunchKpar<128, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.exact_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 256 * 100 + 2:
                return sweepLaunchKpar<256, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.exact_kb,
                    tuning.force_two_phase, gemv_ctx, cuda_device_id, stream);
            case 256 * 100 + 4:
                return sweepLaunchKpar<256, 4, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.exact_kb,
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
        int exact_kb,
        int force_two_phase,
        CUDAGemvContext_ *gemv_ctx,
        int device_id, cudaStream_t stream);

    template <uint8_t CB>
    bool dispatchWideSmallMSerialRowsGenerated(
        const GeneratedDispatchTuning &tuning,
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        cudaStream_t stream)
    {
        g_last_effective_kb = 1;
        switch (tuning.tile_n * 100 + tuning.cpt)
        {
        case 32 * 100 + 1:
        {
            const int grid_n = (N + 32 - 1) / 32;
            nativeVnniGemv_wide_small_m_serial_rows<32, 1, CB><<<dim3(grid_n, M), 32, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        case 64 * 100 + 1:
        {
            const int grid_n = (N + 64 - 1) / 64;
            nativeVnniGemv_wide_small_m_serial_rows<64, 1, CB><<<dim3(grid_n, M), 64, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        case 64 * 100 + 2:
        {
            const int grid_n = (N + 64 - 1) / 64;
            nativeVnniGemv_wide_small_m_serial_rows<64, 2, CB><<<dim3(grid_n, M), 32, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        case 128 * 100 + 1:
        {
            const int grid_n = (N + 128 - 1) / 128;
            nativeVnniGemv_wide_small_m_serial_rows<128, 1, CB><<<dim3(grid_n, M), 128, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        case 128 * 100 + 2:
        {
            const int grid_n = (N + 128 - 1) / 128;
            nativeVnniGemv_wide_small_m_serial_rows<128, 2, CB><<<dim3(grid_n, M), 64, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        case 256 * 100 + 2:
        {
            const int grid_n = (N + 256 - 1) / 256;
            nativeVnniGemv_wide_small_m_serial_rows<256, 2, CB><<<dim3(grid_n, M), 128, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        case 256 * 100 + 4:
        {
            const int grid_n = (N + 256 - 1) / 256;
            nativeVnniGemv_wide_small_m_serial_rows<256, 4, CB><<<dim3(grid_n, M), 64, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        case 512 * 100 + 4:
        {
            const int grid_n = (N + 512 - 1) / 512;
            nativeVnniGemv_wide_small_m_serial_rows<512, 4, CB><<<dim3(grid_n, M), 128, 0, stream>>>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias);
            return cudaGetLastError() == cudaSuccess;
        }
        default:
            return false;
        }
    }

    template <uint8_t CB>
    bool dispatchKparSmallMGenerated(
        const GeneratedDispatchTuning &tuning,
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        int cuda_device_id, cudaStream_t stream)
    {
        switch (tuning.tile_n * 100 + tuning.cpt)
        {
        case 32 * 100 + 1:
            return launchKparSmallMImpl<32, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 1:
            return launchKparSmallMImpl<64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 2:
            return launchKparSmallMImpl<64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 1:
            return launchKparSmallMImpl<128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 2:
            return launchKparSmallMImpl<128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 256 * 100 + 2:
            return launchKparSmallMImpl<256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 256 * 100 + 4:
            /*
             * Several large-K generated M=1 policies, including IQ2_S and
             * IQ2_XXS at `N=1024, K=5120`, use 256 output columns with four
             * columns per thread.  The grouped kernel supports the same
             * 64-thread tile natively; omitting this dispatch case forced an
             * otherwise valid decode-equivalent projection to fail.
             */
            return launchKparSmallMImpl<256, 4, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
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
        bool graph_captured = false;
        if (!queryGraphCapturedExecution(stream, graph_captured))
            return false;

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
                g_sweep.exact_kb,
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
                1,
                N,
                K,
                rm_slot && *rm_slot && (*rm_slot)->d_payload,
                cuda_device_id,
                graph_captured,
                false,
                gemv_ctx);

            return dispatchGeneratedTuning<CB>(
                shape, tuning,
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                gemv_ctx, rm_slot ? *rm_slot : nullptr,
                cuda_device_id, stream);
        }

        /*
         * The generated policy is the complete production authority. A miss is
         * an unsupported shape, not permission to manufacture a default KPAR
         * route or lazily promote to an unmeasured ROWPAR representation.
         */
        NativeGemvShape shape = NativeGemvShape::KPAR;
        GeneratedDispatchTuning tuning{};
        if (!selectGeneratedDispatch<CB>(
                graph_captured, 1, N, K, shape, tuning))
            return false;

        recordGemvDispatch<CB>(
            shape,
            tuning,
            1,
            N,
            K,
            rm_slot && *rm_slot && (*rm_slot)->d_payload,
            cuda_device_id,
            graph_captured,
            false,
            gemv_ctx);

        return dispatchGeneratedTuning<CB>(
            shape, tuning,
            d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
            d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
            gemv_ctx, rm_slot ? *rm_slot : nullptr,
            cuda_device_id, stream);
    }

    template <uint8_t CB>
    bool dispatchCodebookSmallMRowPar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        CUDARowMajorWeights_ **rm_slot,
        int cuda_device_id, cudaStream_t stream)
    {
        bool graph_captured = false;
        if (!queryGraphCapturedExecution(stream, graph_captured))
            return false;

        g_last_effective_kb = 1;
        NativeGemvShape shape = NativeGemvShape::KPAR;
        GeneratedDispatchTuning tuning{};

        if (g_sweep.active)
        {
            /**
             * The generated-dispatch trainer drives this path for verifier
             * runtime-M buckets. Keep the sweep override local to backend
             * tuning: production dispatch still comes from the generated
             * policy below, while the perf harness can time real KPAR
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
                g_sweep.exact_kb,
            };
        }
        else
        {
            /**
             * Decode-equivalent verifier publication is stricter than ordinary
             * small-M GEMV parity.  When the scoped M=1 verifier flag is active,
             * classify by the serial M=1 route and launch a grouped row-dimension
             * kernel for that family.  This preserves the target architecture
             * (one grouped production operation) while matching the family that
             * serial decode uses for each accepted row.
             */
            const bool decode_equivalent_m1 = decodeEquivalentM1ConfigActive();
            const int dispatch_m =
                decode_equivalent_m1 ? 1 : (llaminar2::debugEnv().gemm.deterministic ? 2 : M);
            if (!selectGeneratedDispatch<CB>(
                    graph_captured, dispatch_m, N, K, shape, tuning))
            {
                return false;
            }
        }

        const bool decode_equivalent_m1 = decodeEquivalentM1ConfigActive();
        recordGemvDispatch<CB>(
            shape,
            tuning,
            M,
            N,
            K,
            rm_slot && *rm_slot && (*rm_slot)->d_payload,
            cuda_device_id,
            graph_captured,
            decode_equivalent_m1,
            gemv_ctx);
        if (shape == NativeGemvShape::WIDE || shape == NativeGemvShape::DIRECT)
        {
            if (!decode_equivalent_m1)
                return false;
            return dispatchWideSmallMSerialRowsGenerated<CB>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                stream);
        }

        if constexpr (CB != 19)
        {
            /* Explicit diagnostic ROWPAR candidates own their preparation. */
            if (shape == NativeGemvShape::ROWPAR && rm_slot)
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
                    return launchRowParSmallM<CB>(
                        d_A_int8, (*rm_slot)->d_payload, (*rm_slot)->d_scales,
                        (*rm_slot)->d_mins, (*rm_slot)->d_emins, d_C,
                        d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                        tuning.tile_n, stream);
                }

                cudaGetLastError();
                return false;
            }
        }

        if (shape == NativeGemvShape::KPAR)
        {
            return dispatchKparSmallMGenerated<CB>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                gemv_ctx, cuda_device_id, stream);
        }

        return false;
    }

    bool dispatchSmallMByCodebook(
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
        uint8_t codebook_id,
        int cuda_device_id,
        cudaStream_t cuda_stream,
        CUDAGemvContext_ *gemv_ctx,
        CUDARowMajorWeights_ **rm_slot)
    {
        switch (codebook_id)
        {
        case 0:
            return dispatchCodebookSmallMRowPar<0>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 4:
            return dispatchCodebookSmallMRowPar<4>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 5:
            return dispatchCodebookSmallMRowPar<5>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 6:
            return dispatchCodebookSmallMRowPar<6>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 7:
            return dispatchCodebookSmallMRowPar<7>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 8:
            return dispatchCodebookSmallMRowPar<8>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 9:
            return dispatchCodebookSmallMRowPar<9>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 10:
            return dispatchCodebookSmallMRowPar<10>(d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 11:
            return dispatchCodebookSmallMRowPar<11>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 12:
            return dispatchCodebookSmallMRowPar<12>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 13:
            return dispatchCodebookSmallMRowPar<13>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 14:
            return dispatchCodebookSmallMRowPar<14>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 15:
            return dispatchCodebookSmallMRowPar<15>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 16:
            return dispatchCodebookSmallMRowPar<16>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 17:
            return dispatchCodebookSmallMRowPar<17>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
        case 19:
            return dispatchCodebookSmallMRowPar<19>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, M, N, K, alpha, beta, d_C_existing, d_bias, gemv_ctx, rm_slot, cuda_device_id, cuda_stream);
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
        int exact_kb,
        int force_two_phase,
        CUDAGemvContext_ *gemv_ctx,
        int device_id, cudaStream_t stream)
    {
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const bool debug_deterministic = llaminar2::debugEnv().gemm.deterministic;
        const bool verifier_decode_equivalent = decodeEquivalentM1ConfigActive();
        const int kb_capped = resolveKparKBlocks(
            grid_n, k_groups, num_sms,
            target_waves, min_kgroups_per_cta, max_kb, exact_kb);
        if (kb_capped <= 0)
            return false;
        g_last_effective_kb = kb_capped;

        const size_t partials_bytes = static_cast<size_t>(kb_capped) * N * sizeof(float);
        bool use_two_phase;
        if (debug_deterministic || verifier_decode_equivalent)
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
        if (M < 2 || N <= 0 || K <= 0 || (K % BLOCK_K) != 0)
            return false;
        if (!cudaNativeVNNIGemvTuned_supportsCodebook(codebook_id))
            return false;
        if (!gemv_ctx)
            return false;

        cudaError_t err = cudaSetDevice(cuda_device_id);
        if (err != cudaSuccess)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        return dispatchSmallMByCodebook(
            d_A_int8, d_payload, d_scales, d_mins, d_emins,
            d_C_fp32, d_scales_A_block, M, N, K, alpha, beta,
            d_C_existing, d_bias, codebook_id, cuda_device_id,
            cuda_stream, gemv_ctx, rm_slot);
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
    // exact_kb: 0=generated occupancy policy, >0=one exact trainer schedule
    // force_two_phase: 0=auto, 1=force two-phase, 2=force atomic
    // =====================================================================
    void cudaNativeVNNIGemvSweep_setConfig(
        int kernel_family, int tile_n, int cpt,
        int target_waves, int mkg, int max_kb,
        int exact_kb,
        int force_two_phase)
    {
        g_sweep.active = true;
        g_sweep.kernel_family = kernel_family;
        g_sweep.tile_n = tile_n;
        g_sweep.cpt = cpt;
        g_sweep.target_waves = target_waves;
        g_sweep.mkg = mkg;
        g_sweep.max_kb = max_kb;
        g_sweep.exact_kb = exact_kb;
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

    int cudaNativeVNNIGemvTuned_getLastEffectiveKBlocks()
    {
        return g_last_effective_kb;
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
        case 6:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 6>(d_payload_col, N, K_blocks, s));
            break;
        case 8:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 8>(d_payload_col, N, K_blocks, s));
            break;
        case 9:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 9>(d_payload_col, N, K_blocks, s));
            break;
        case 12:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 12>(d_payload_col, N, K_blocks, s));
            break;
        case 13:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 13>(d_payload_col, N, K_blocks, s));
            break;
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
        case 20:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 20>(d_payload_col, N, K_blocks, s));
            break;
        case 24:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 24>(d_payload_col, N, K_blocks, s));
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

extern "C" void cudaNativeVNNIGemvTuned_setDecodeEquivalentM1Config(int enabled)
{
    g_cuda_native_vnni_decode_equivalent_m1_config = enabled ? 1 : 0;
}

extern "C" int cudaNativeVNNIGemvTuned_getDecodeEquivalentM1Config()
{
    return g_cuda_native_vnni_decode_equivalent_m1_config;
}
