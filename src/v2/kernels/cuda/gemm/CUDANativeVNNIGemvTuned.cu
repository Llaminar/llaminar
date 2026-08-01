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
#include "kernels/common/NativeVNNIDispatchCache.h"
#include "backends/BackendManager.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"
#include "utils/PrefillGraphBucketDefaults.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <limits>
#include <mutex>

static thread_local int g_cuda_native_vnni_decode_equivalent_m1_config = 0;
static thread_local int g_cuda_native_vnni_serial_partition_n = 0;

static bool decodeEquivalentM1ConfigActive()
{
    return g_cuda_native_vnni_decode_equivalent_m1_config != 0;
}

static int serialEquivalentPolicyN(int actual_n)
{
    return g_cuda_native_vnni_serial_partition_n > 0
               ? g_cuda_native_vnni_serial_partition_n
               : actual_n;
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

/**
 * @brief Clamp a grouped K-part tile to the persistent bound workspace.
 *
 * Workspace planning deliberately reserves a stable verifier tile rather than
 * scaling scratch with prompt M. Extended verifier depths remain total by
 * launching multiple grouped tiles against that arena. A multi-row operation
 * must retain room for at least two rows; returning zero hard-fails an
 * under-provisioned binding instead of degenerating into production row replay.
 */
static int workspaceBoundedKparTileRows(
    const CUDAGemvContext_ *ctx,
    int requested_rows,
    int k_partitions,
    int n)
{
    if (!ctx || !ctx->kpar_partials || requested_rows <= 0 ||
        k_partitions <= 0 || n <= 0)
    {
        return 0;
    }

    const size_t floats_per_row =
        static_cast<size_t>(k_partitions) * static_cast<size_t>(n);
    if (floats_per_row == 0)
        return 0;

    const size_t capacity_rows = ctx->kpar_capacity / floats_per_row;
    const size_t required_group_rows = requested_rows > 1 ? size_t{2} : size_t{1};
    if (capacity_rows < required_group_rows)
        return 0;

    return std::min(
        requested_rows,
        static_cast<int>(std::min(
            capacity_rows,
            static_cast<size_t>(std::numeric_limits<int>::max()))));
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
    /** Trainer-only grouped row-reuse override; zero selects production policy. */
    static int g_grouped_rows_override = 0;
    /** Trainer-only integer tensor-core grouped candidate selector. */
    static thread_local bool g_grouped_tensor_core_override = false;
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

    /** Complete host identity for one generated CUDA decode lookup. */
    using CUDAGeneratedDispatchCacheKey = std::array<uint64_t, 4>;

    /** Cached generated result; selector misses remain explicit in the cache. */
    struct CUDAGeneratedDispatchSelection
    {
        NativeGemvShape shape = NativeGemvShape::KPAR;
        GeneratedDispatchTuning tuning{};
    };

    /**
     * @brief Resolve immutable CUDA decode policy through a thread-local cache.
     *
     * Keep the untruncated M/N/K values in the key. The generated exact ABI
     * packs M into seven bits, but an unsupported M outside that range must not
     * alias a supported exact overlay. The codebook is a template parameter,
     * so each physical format owns an independent cache without a runtime
     * branch in the production launcher.
     */
    template <uint8_t CB>
    bool selectCachedGeneratedDispatch(
        bool graph_captured,
        int m,
        int n,
        int k,
        NativeGemvShape &shape,
        GeneratedDispatchTuning &tuning)
    {
        const int policy_n = serialEquivalentPolicyN(n);
        const CUDAGeneratedDispatchCacheKey key = {
            graph_captured ? 1ULL : 0ULL,
            static_cast<uint64_t>(static_cast<uint32_t>(m)),
            static_cast<uint64_t>(static_cast<uint32_t>(policy_n)),
            static_cast<uint64_t>(static_cast<uint32_t>(k)),
        };
        using Cache = llaminar2::native_vnni::FixedDispatchCache<
            CUDAGeneratedDispatchCacheKey,
            CUDAGeneratedDispatchSelection,
            64,
            llaminar2::native_vnni::DispatchCacheArrayHasher<4>>;
        static thread_local Cache cache;

        bool selected = false;
        if (const CUDAGeneratedDispatchSelection *cached =
                cache.lookupValue(key, selected))
        {
            shape = cached->shape;
            tuning = cached->tuning;
            return selected;
        }

        CUDAGeneratedDispatchSelection cached{};
        selected = selectGeneratedDispatch<CB>(
            graph_captured, m, policy_n, k, cached.shape, cached.tuning);
        cache.insert(key, selected, cached);
        shape = cached.shape;
        tuning = cached.tuning;
        return selected;
    }

#if defined(LLAMINAR_CUDA_GROUPED_DISPATCH_POLICY_V2)
    /**
     * @brief Resolve the grouped-verifier policy with the same cache contract.
     */
    template <uint8_t CB>
    bool selectCachedGeneratedGroupedTuning(
        bool graph_captured,
        int m,
        int n,
        int k,
        GeneratedGroupedTuning &tuning)
    {
        const int policy_n = serialEquivalentPolicyN(n);
        const CUDAGeneratedDispatchCacheKey key = {
            graph_captured ? 1ULL : 0ULL,
            static_cast<uint64_t>(static_cast<uint32_t>(m)),
            static_cast<uint64_t>(static_cast<uint32_t>(policy_n)),
            static_cast<uint64_t>(static_cast<uint32_t>(k)),
        };
        using Cache = llaminar2::native_vnni::FixedDispatchCache<
            CUDAGeneratedDispatchCacheKey,
            GeneratedGroupedTuning,
            64,
            llaminar2::native_vnni::DispatchCacheArrayHasher<4>>;
        static thread_local Cache cache;

        bool selected = false;
        GeneratedGroupedTuning cached{};
        if (cache.lookup(key, selected, cached))
        {
            tuning = cached;
            return selected;
        }

        selected = selectGeneratedGroupedTuning<CB>(
            graph_captured, m, policy_n, k, cached);
        cache.insert(key, selected, cached);
        tuning = cached;
        return selected;
    }
#endif

    /** @brief Runtime-codebook adapter used by host-only cache perf tests. */
    bool queryCachedGeneratedDispatch(
        uint8_t codebook_id,
        bool graph_captured,
        int m,
        int n,
        int k,
        NativeGemvShape &shape,
        GeneratedDispatchTuning &tuning)
    {
#define LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(CODEBOOK) \
        case CODEBOOK: \
            return selectCachedGeneratedDispatch<CODEBOOK>( \
                graph_captured, m, n, k, shape, tuning)
        switch (codebook_id)
        {
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(0);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(4);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(5);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(6);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(7);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(8);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(9);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(10);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(11);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(12);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(13);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(14);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(15);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(16);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(17);
            LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK(19);
        default:
            return false;
        }
#undef LLAMINAR_QUERY_CUDA_NVNNI_CODEBOOK
    }

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
    __device__ __forceinline__ float native_vnni_block_contribution_from_dots_rn(
        const int32_t *__restrict__ a_vals,
        int dot_lo,
        int dot_hi,
        const uint8_t *__restrict__ payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        size_t linear,
        float scale_a)
    {
        using Traits =
            llaminar2::cuda_native_vnni::CodebookTraits<CB>;

        int sum_a = 0;
        int sum_lo = 0;
        int sum_hi = 0;
        int subgroup_sums[4] = {0, 0, 0, 0};

#pragma unroll
        for (int group = 0; group < 8; ++group)
        {
            const int group_sum =
                llaminar2::cuda_native_vnni::sum_packed_i8(
                    a_vals[group]);
            sum_a += group_sum;
            if (group < 4)
                sum_lo += group_sum;
            else
                sum_hi += group_sum;
            subgroup_sums[group / 2] += group_sum;
        }

        const uint16_t secondary_bits =
            d_mins ? d_mins[linear] : uint16_t{0};
        const uint32_t emin_bits =
            d_emins ? d_emins[linear] : uint32_t{0};
        const uint16_t iq1m_qh =
            Traits::is_iq1_m
                ? static_cast<uint16_t>(payload[4]) |
                      (static_cast<uint16_t>(payload[5]) << 8)
                : uint16_t{0};

        return llaminar2::cuda_native_vnni::
            native_vnni_block_contribution_from_reduced_terms_rn<CB>(
                dot_lo,
                dot_hi,
                scale_a,
                d_scales[linear],
                secondary_bits,
                emin_bits,
                sum_a,
                sum_lo,
                sum_hi,
                iq1m_qh,
                subgroup_sums[0],
                subgroup_sums[1],
                subgroup_sums[2],
                subgroup_sums[3]);
    }

    /**
     * @brief Decode one block with DP4A, then publish through the shared exact
     *        FP32 contribution contract.
     *
     * The grouped tensor-core candidate below supplies mathematically equal
     * integer dot products to the same helper. Keeping scale, asymmetric-min,
     * and IQ1 delta arithmetic here prevents the two execution engines from
     * acquiring different compiler contraction or operand-order behavior.
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
        int dot_lo = 0;
        int dot_hi = 0;
        if constexpr (llaminar2::cuda_native_vnni::CodebookTraits<CB>::is_dual_scale)
        {
#pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                dot_lo = __dp4a(a_vals[g], packed_groups[g], dot_lo);
                dot_hi = __dp4a(a_vals[g + 4], packed_groups[g + 4], dot_hi);
            }
        }
        else
        {
#pragma unroll
            for (int g = 0; g < 8; ++g)
                dot_lo = __dp4a(a_vals[g], packed_groups[g], dot_lo);
        }

        return native_vnni_block_contribution_from_dots_rn<CB>(
            a_vals,
            dot_lo,
            dot_hi,
            payload,
            d_scales,
            d_mins,
            d_emins,
            linear,
            scale_a);
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
    static T *transposeBuffer(
        const T *d_col,
        int N,
        int K_blocks,
        int cuda_device_id,
        cudaStream_t stream)
    {
        const size_t total_elements = static_cast<size_t>(N) * K_blocks;
        const size_t total_bytes = total_elements * ELEM_BYTES;

        auto *const backend = llaminar2::getCUDABackend();
        if (!backend)
            return nullptr;
        T *d_row = static_cast<T *>(
            backend->allocate(total_bytes, cuda_device_id));
        if (!d_row)
            return nullptr;

        const int threads = 256;
        const int blocks = (static_cast<int>(total_elements) + threads - 1) / threads;
        transpose_blocks_kernel<ELEM_BYTES><<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t *>(d_col),
            reinterpret_cast<uint8_t *>(d_row),
            N, K_blocks);

        if (cudaGetLastError() != cudaSuccess)
        {
            backend->free(d_row, cuda_device_id);
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
     * @brief Canonical generated-policy surface for production decode.
     *
     * The decode graph state machine necessarily executes a new shape once as
     * an eager warmup before recording and replaying its CUDA graph. Dispatch
     * mode therefore cannot be allowed to change the reduction tree: otherwise
     * an MTP verifier warmup can use a different K partition from an already
     * captured serial M=1 oracle. The graph-captured policy is the steady-state
     * performance authority, so eager warmup deliberately selects that same
     * family, tile, and exact K partition.
     */
    static constexpr bool kCanonicalDecodePolicyGraphCaptured = true;

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
        bool verifier_serial_m1,
        int grouped_rows)
    {
        if (verifier_serial_m1 && M >= 2)
        {
            std::string candidate =
                "cuda.nvnni.decode.verifier.inherit_serial_m1";
            const int effective_grouped_rows = g_grouped_rows_override > 0
                                                   ? g_grouped_rows_override
                                                   : grouped_rows;
            if (effective_grouped_rows > 0)
            {
                candidate += ".r" +
                             std::to_string(effective_grouped_rows);
            }
            return candidate;
        }

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
        CUDAGemvContext_ *gemv_ctx,
        int grouped_rows = 0,
        int policy_graph_captured = -1)
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
                {"policy_execution_mode",
                 (policy_graph_captured < 0
                      ? graph_captured
                      : policy_graph_captured != 0)
                     ? "graph_captured"
                     : "eager"},
                {"semantic_contract", verifier_serial_m1
                                             ? "verifier_serial_m1_bitwise"
                                             : "fast"},
                {"effective_candidate_id",
                 effectiveCandidateName(
                     shape, tuning, effective_kb, M, verifier_serial_m1,
                     grouped_rows)},
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

    /** Record one byte-gated tensor-core verifier candidate invocation. */
    template <uint8_t CB>
    static void recordTensorCoreVerifierDispatch(
        const GeneratedDispatchTuning &serial_tuning,
        int M,
        int N,
        int K,
        int effective_kb,
        int cuda_device_id,
        bool graph_captured)
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
                {"m", std::to_string(M)},
                {"n", std::to_string(N)},
                {"k", std::to_string(K)},
                {"execution_mode", graph_captured
                                       ? "graph_captured"
                                       : "eager"},
                {"semantic_contract", "verifier_serial_m1_bitwise"},
                {"effective_candidate_id",
                 "cuda.nvnni.decode.verifier.tensor_core_mma16"},
                {"route", "tensor_core"},
                {"tile_n", "8"},
                {"cpt", "0"},
                {"effective_kb", std::to_string(effective_kb)},
                {"target_waves", std::to_string(serial_tuning.target_waves)},
                {"mkg", std::to_string(serial_tuning.mkg)},
                {"max_kb", std::to_string(serial_tuning.max_kb)},
                {"exact_kb", std::to_string(serial_tuning.exact_kb)},
                {"force_two_phase", "1"},
                {"rowmajor_available", "false"}});
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

        const int blocks_per_row = K / BLOCK_K;

        // Shared memory: 8 int32_t = 32 bytes for the A activation block
        __shared__ int32_t smem_A[8];

        float acc[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[c] = 0.0f;

        for (int blk = 0; blk < blocks_per_row; ++blk)
        {
            /*
             * Every thread must remain alive through both CTA barriers, even
             * on the final partially populated N tile. Returning threads with
             * n_base >= N here leaves the valid tail threads waiting at a
             * CTA-wide barrier and permits them to consume undefined shared
             * activation state. Invalid columns simply skip their arithmetic
             * and final publication below.
             */
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

    /** Return the row owned by one `m16n8k32` accumulator element. */
    __device__ __forceinline__ int verifierMmaFragmentRow(int lane, int element)
    {
        return (element >> 1) * 8 + (lane >> 2);
    }

    /** Return the column owned by one `m16n8k32` accumulator element. */
    __device__ __forceinline__ int verifierMmaFragmentColumn(
        int lane,
        int element)
    {
        return (lane & 3) * 2 + (element & 1);
    }

    /** Load one row-major 16x32 signed-int8 A fragment from shared memory. */
    __device__ __forceinline__ void loadVerifierMmaA(
        uint32_t fragment[4],
        const int *shared_base,
        int lane)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int kStrideWords = 8;
        const int *address = shared_base +
                             (lane % 16) * kStrideWords +
                             (lane / 16) * 4;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(fragment[0]), "=r"(fragment[1]),
              "=r"(fragment[2]), "=r"(fragment[3])
            : "l"(address));
#else
        (void)fragment;
        (void)shared_base;
        (void)lane;
#endif
    }

    /** Load one column-major 32x8 signed-int8 B fragment from shared memory. */
    __device__ __forceinline__ void loadVerifierMmaB(
        uint32_t fragment[2],
        const int *shared_base,
        int lane)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int kStrideWords = 8;
        const int *address = shared_base +
                             (lane % 8) * kStrideWords +
                             ((lane / 8) * 4) % 8;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
            : "=r"(fragment[0]), "=r"(fragment[1])
            : "l"(address));
#else
        (void)fragment;
        (void)shared_base;
        (void)lane;
#endif
    }

    /** Execute one exact signed-int8 `m16n8k32` integer MMA. */
    __device__ __forceinline__ void verifierMmaM16N8K32(
        int32_t accumulator[4],
        const uint32_t a_fragment[4],
        const uint32_t b_fragment[2])
    {
#if __CUDA_ARCH__ >= 800
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3},"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%0, %1, %2, %3};\n"
            : "+r"(accumulator[0]), "+r"(accumulator[1]),
              "+r"(accumulator[2]), "+r"(accumulator[3])
            : "r"(a_fragment[0]), "r"(a_fragment[1]),
              "r"(a_fragment[2]), "r"(a_fragment[3]),
              "r"(b_fragment[0]), "r"(b_fragment[1]));
#else
        (void)accumulator;
        (void)a_fragment;
        (void)b_fragment;
#endif
    }

    /**
     * @brief Batch-invariant grouped verifier kernel using integer tensor cores.
     *
     * One warp owns a 16-row by 8-column output tile and one public-M1 K
     * partition. Every 32-value NativeVNNI block is decoded once per output
     * column, multiplied as exact INT8xINT8->INT32 MMA, and then converted into
     * FP32 by `native_vnni_block_contribution_from_dots_rn`. Four independent
     * warps share one CTA only to avoid Ampere's low blocks-per-SM occupancy
     * ceiling; each warp owns private shared-memory slices and performs no
     * cross-warp arithmetic or synchronization. That is the same
     * explicit scale/min/delta and ascending block accumulation contract used
     * by serial DP4A decode. Runtime M only changes the independent Y-grid tile
     * count; it cannot alter an accepted row's arithmetic.
     */
    template <uint8_t CB>
    __global__ __launch_bounds__(128, 8) void nativeVnniGemvTensorCoreSmallM(
        const int8_t *__restrict__ d_A_int8,
        const uint8_t *__restrict__ d_payload,
        const uint16_t *__restrict__ d_scales,
        const uint16_t *__restrict__ d_mins,
        const uint32_t *__restrict__ d_emins,
        float *__restrict__ d_partials,
        const float *__restrict__ d_scales_A,
        int M,
        int N,
        int K,
        int kb,
        float alpha)
    {
#if __CUDA_ARCH__ >= 800
        constexpr int kRows = 16;
        constexpr int kColumns = 8;
        constexpr int kWarpsPerBlock = 4;
        constexpr int kWordsPerBlock = BLOCK_K / sizeof(int32_t);
        const int warp = threadIdx.x / warpSize;
        const int lane = threadIdx.x % warpSize;
        const int row_base = blockIdx.y * kRows;
        const int column_base =
            (blockIdx.x * kWarpsPerBlock + warp) * kColumns;
        if (column_base >= N)
            return;
        const int split_index = blockIdx.z;
        const int blocks_per_row = K / BLOCK_K;
        const int blocks_per_split =
            (blocks_per_row + kb - 1) / kb;
        const int block_begin = split_index * blocks_per_split;
        const int block_end = min(
            blocks_per_row,
            block_begin + blocks_per_split);

        __shared__ __align__(16) int8_t
            shared_a[kWarpsPerBlock][kRows * BLOCK_K];
        __shared__ __align__(16) int8_t
            shared_b[kWarpsPerBlock][kColumns * BLOCK_K];
        int8_t *warp_a = shared_a[warp];
        int8_t *warp_b = shared_b[warp];

        float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int block = block_begin; block < block_end; ++block)
        {
            /* Exactly one 16-byte A vector load per lane fills 16x32 bytes. */
            const int local_row = lane >> 1;
            const int half = lane & 1;
            const int global_row = row_base + local_row;
            int4 activation = make_int4(0, 0, 0, 0);
            if (global_row < M)
            {
                activation = *reinterpret_cast<const int4 *>(
                    d_A_int8 + static_cast<size_t>(global_row) * K +
                    block * BLOCK_K + half * sizeof(int4));
            }
            *reinterpret_cast<int4 *>(
                warp_a + local_row * BLOCK_K + half * sizeof(int4)) =
                activation;

            /* Eight lanes decode the eight output columns once per K block. */
            if (lane < kColumns)
            {
                int32_t packed_groups[kWordsPerBlock];
                const int column = column_base + lane;
                if (column < N)
                {
                    const size_t linear =
                        static_cast<size_t>(block) * N + column;
                    const uint8_t *payload = d_payload +
                        linear * llaminar2::cuda_native_vnni::
                                     payload_bytes_for_codebook<CB>();
                    llaminar2::cuda_native_vnni::decode_groups_vec<CB>(
                        payload,
                        packed_groups);
                }
                else
                {
#pragma unroll
                    for (int group = 0; group < kWordsPerBlock; ++group)
                        packed_groups[group] = 0;
                }
                *reinterpret_cast<int4 *>(warp_b + lane * BLOCK_K) =
                    make_int4(
                        packed_groups[0],
                        packed_groups[1],
                        packed_groups[2],
                        packed_groups[3]);
                *reinterpret_cast<int4 *>(
                    warp_b + lane * BLOCK_K + sizeof(int4)) =
                    make_int4(
                        packed_groups[4],
                        packed_groups[5],
                        packed_groups[6],
                        packed_groups[7]);
            }
            __syncwarp();

            uint32_t a_fragment[4];
            uint32_t b_fragment[2];
            loadVerifierMmaA(
                a_fragment,
                reinterpret_cast<const int *>(warp_a),
                lane);
            loadVerifierMmaB(
                b_fragment,
                reinterpret_cast<const int *>(warp_b),
                lane);

            int32_t dot_lo[4] = {0, 0, 0, 0};
            int32_t dot_hi[4] = {0, 0, 0, 0};
            if constexpr (llaminar2::cuda_native_vnni::
                              CodebookTraits<CB>::is_dual_scale)
            {
                const uint32_t b_lo[2] = {b_fragment[0], 0u};
                const uint32_t b_hi[2] = {0u, b_fragment[1]};
                verifierMmaM16N8K32(dot_lo, a_fragment, b_lo);
                verifierMmaM16N8K32(dot_hi, a_fragment, b_hi);
            }
            else
            {
                verifierMmaM16N8K32(dot_lo, a_fragment, b_fragment);
            }

#pragma unroll
            for (int element = 0; element < 4; ++element)
            {
                const int row = row_base +
                    verifierMmaFragmentRow(lane, element);
                const int column = column_base +
                    verifierMmaFragmentColumn(lane, element);
                if (row >= M || column >= N)
                    continue;

                const int local_fragment_row =
                    verifierMmaFragmentRow(lane, element);
                const int32_t *activation_words =
                    reinterpret_cast<const int32_t *>(
                        warp_a + local_fragment_row * BLOCK_K);
                const size_t linear =
                    static_cast<size_t>(block) * N + column;
                const uint8_t *payload = d_payload +
                    linear * llaminar2::cuda_native_vnni::
                                 payload_bytes_for_codebook<CB>();
                const float scale_a = d_scales_A[
                    static_cast<size_t>(row) * blocks_per_row + block];
                const float contribution =
                    native_vnni_block_contribution_from_dots_rn<CB>(
                        activation_words,
                        dot_lo[element],
                        dot_hi[element],
                        payload,
                        d_scales,
                        d_mins,
                        d_emins,
                        linear,
                        scale_a);
                accumulator[element] = __fadd_rn(
                    accumulator[element],
                    contribution);
            }
            __syncwarp();
        }

#pragma unroll
        for (int element = 0; element < 4; ++element)
        {
            const int row = row_base +
                verifierMmaFragmentRow(lane, element);
            const int column = column_base +
                verifierMmaFragmentColumn(lane, element);
            if (row < M && column < N)
            {
                d_partials[
                    (static_cast<size_t>(split_index) * M + row) * N +
                    column] = __fmul_rn(alpha, accumulator[element]);
            }
        }
#else
        (void)d_A_int8;
        (void)d_payload;
        (void)d_scales;
        (void)d_mins;
        (void)d_emins;
        (void)d_partials;
        (void)d_scales_A;
        (void)M;
        (void)N;
        (void)K;
        (void)kb;
        (void)alpha;
#endif
    }

    /**
     * @brief Compile-time physical launch geometry for grouped DP4A tiles.
     *
     * One logical N tile has no shared state or synchronization with another,
     * so low-register row depths can pack several tiles into one CTA. This
     * avoids the one-warp-block occupancy ceiling without changing the number,
     * order, or ownership of any dot product. Deeper row tiles retain smaller
     * blocks so their larger accumulator arrays remain launchable without
     * register spilling.
     */
    template <int ROWS_PER_TILE, int TILE_N, int CPT>
    struct GroupedDp4aLaunchGeometry
    {
        static constexpr int threads_per_n_tile = TILE_N / CPT;
        static constexpr int n_tiles_per_block =
            threads_per_n_tile == 32
                ? (ROWS_PER_TILE <= 8 ? 4 : (ROWS_PER_TILE <= 16 ? 2 : 1))
                : (threads_per_n_tile == 64 && ROWS_PER_TILE <= 8 ? 2 : 1);
        static constexpr int threads_per_block =
            threads_per_n_tile * n_tiles_per_block;
        static constexpr int columns_per_block =
            TILE_N * n_tiles_per_block;
    };

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
        using Geometry = GroupedDp4aLaunchGeometry<ROWS_PER_TILE, TILE_N, CPT>;
        const int n_tile_in_block =
            threadIdx.x / Geometry::threads_per_n_tile;
        const int n_tile_thread =
            threadIdx.x % Geometry::threads_per_n_tile;
        const int n_tile =
            static_cast<int>(blockIdx.x) * Geometry::n_tiles_per_block +
            n_tile_in_block;
        const int n_base = n_tile * TILE_N + n_tile_thread * CPT;
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
            /*
             * Decode each owned output column once, then stream verifier rows
             * through those decoded weights. The previous implementation kept
             * eight packed activation words for every row live at once. That
             * consumed 130 registers/thread at eight-row reuse and 231 at
             * sixteen-row reuse on SM86, collapsing occupancy even though no
             * values spilled. This transposition keeps only one row's eight
             * activation words live while retaining weight reuse across every
             * row and activation reuse across every CPT column.
             */
            int32_t packed_groups[CPT][8];
            const uint8_t *payloads[CPT];
            size_t linears[CPT];
#pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n >= N)
                    break;
                const size_t linear = static_cast<size_t>(blk) * N + n;
                const uint8_t *payload =
                    d_payload + linear *
                                    llaminar2::cuda_native_vnni::
                                        payload_bytes_for_codebook<CB>();
                linears[c] = linear;
                payloads[c] = payload;
                llaminar2::cuda_native_vnni::decode_groups_vec<CB>(
                    payload,
                    packed_groups[c]);
            }

#pragma unroll
            for (int tile_row = 0; tile_row < ROWS_PER_TILE; ++tile_row)
            {
                const int row = row_base + tile_row;
                if (row >= M)
                    continue;

                int32_t a_vals[8];
                const int4 *a_ptr128 = reinterpret_cast<const int4 *>(
                    d_A_int8 + static_cast<size_t>(row) * K + blk * BLOCK_K);
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

                const float scale_a =
                    d_scales_A[static_cast<size_t>(row) *
                                   blocks_per_row +
                               blk];

#pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n >= N)
                        break;
                    const float contribution = native_vnni_block_contribution_rn<CB>(
                        a_vals,
                        packed_groups[c],
                        payloads[c],
                        d_scales,
                        d_mins,
                        d_emins,
                        linears[c],
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

    /**
     * @brief Publish a grouped WIDE/DIRECT row from its only K partition.
     *
     * Serial WIDE/DIRECT decode owns one accumulator spanning K and therefore
     * does not add that accumulator to an initialized zero during publication.
     * Loading partition zero directly preserves that exact arithmetic tree and
     * signed-zero behavior while allowing the preceding grouped DP4A kernel to
     * decode every weight block once for several verifier rows.
     */
    __global__ void nativeVnniGemv_publish_single_partition_small_m(
        const float *__restrict__ d_partials,
        float *__restrict__ d_C,
        const float *__restrict__ d_C_existing,
        const float *__restrict__ d_bias,
        int M, int N, float beta)
    {
        const int row = blockIdx.y;
        const int n = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= M || n >= N)
            return;

        const size_t out_idx = static_cast<size_t>(row) * N + n;
        float out = d_partials[out_idx];
        if (beta != 0.0f && d_C_existing)
            out = __fadd_rn(out, __fmul_rn(beta, d_C_existing[out_idx]));
        if (d_bias)
            out = __fadd_rn(out, d_bias[n]);
        d_C[out_idx] = out;
    }

    // =====================================================================
    // KPAR tile profiles — different TILE_N × CPT × K-split combinations
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

    template <
        int TILE_N,
        int CPT,
        uint8_t CB,
        int ROWS_PER_TILE = 2,
        bool MATCH_WIDE_SERIAL = false>
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
        static_assert(
            ROWS_PER_TILE == 2 || ROWS_PER_TILE == 4 ||
                ROWS_PER_TILE == 8 || ROWS_PER_TILE == 16 ||
                ROWS_PER_TILE == 32 || ROWS_PER_TILE == 64,
            "Grouped NativeVNNI row reuse must be a trained instantiation");
        using Geometry = GroupedDp4aLaunchGeometry<ROWS_PER_TILE, TILE_N, CPT>;
        constexpr int THREADS = Geometry::threads_per_block;

        const int serial_grid_n = (N + TILE_N - 1) / TILE_N;
        const int launch_grid_n =
            (N + Geometry::columns_per_block - 1) /
            Geometry::columns_per_block;
        const int k_groups = K / BLOCK_K;
        const int num_sms = querySmCount(gemv_ctx);
        const int kb_capped = MATCH_WIDE_SERIAL
                                  ? 1
                                  : resolveKparKBlocks(
                                        serial_grid_n, k_groups, num_sms,
                                        target_waves, min_kgroups_per_cta,
                                        max_kb, exact_kb);
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
        const int desired_tile_rows =
            llaminar2::nativeVNNIBatchInvariantTileRows(M, N, K);
        const int workspace_tile_rows = workspaceBoundedKparTileRows(
            gemv_ctx,
            desired_tile_rows,
            kb_capped,
            N);
        if (workspace_tile_rows <= 0)
            return false;
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
            const dim3 grid(launch_grid_n, kb_capped, row_tiles);

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
            if constexpr (MATCH_WIDE_SERIAL)
            {
                nativeVnniGemv_publish_single_partition_small_m<<<
                    dim3(reduction_blocks, tile_m), 256, 0, stream>>>(
                    d_partials,
                    tile_c,
                    tile_existing,
                    d_bias,
                    tile_m,
                    N,
                    beta);
            }
            else
            {
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
            }
            if (cudaGetLastError() != cudaSuccess)
                return false;
        }

        return true;
    }

    template <uint8_t CB>
    bool launchTensorCoreSmallMGenerated(
        NativeGemvShape serial_shape,
        const GeneratedDispatchTuning &serial_tuning,
        const int8_t *d_A_int8,
        const uint8_t *d_payload,
        const uint16_t *d_scales,
        const uint16_t *d_mins,
        const uint32_t *d_emins,
        float *d_C,
        const float *d_scales_A,
        int M,
        int N,
        int K,
        float alpha,
        float beta,
        const float *d_C_existing,
        const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        cudaStream_t stream)
    {
        int kb = 1;
        if (serial_shape == NativeGemvShape::KPAR)
        {
            const int serial_grid_n =
                (N + serial_tuning.tile_n - 1) / serial_tuning.tile_n;
            kb = resolveKparKBlocks(
                serial_grid_n,
                K / BLOCK_K,
                querySmCount(gemv_ctx),
                serial_tuning.target_waves,
                serial_tuning.mkg,
                serial_tuning.max_kb,
                serial_tuning.exact_kb);
        }
        if (kb <= 0)
            return false;
        g_last_effective_kb = kb;

        const int desired_tile_rows =
            llaminar2::nativeVNNIBatchInvariantTileRows(M, N, K);
        const int workspace_tile_rows = workspaceBoundedKparTileRows(
            gemv_ctx,
            desired_tile_rows,
            kb,
            N);
        if (workspace_tile_rows <= 0)
            return false;
        float *partials = getKparPartials(
            gemv_ctx,
            static_cast<size_t>(kb) * workspace_tile_rows * N);
        if (!partials)
            return false;

        const int activation_blocks_per_row = K / BLOCK_K;
        for (int row_base = 0; row_base < M;
             row_base += workspace_tile_rows)
        {
            const int tile_m = std::min(
                workspace_tile_rows,
                M - row_base);
            const int8_t *tile_a = d_A_int8 +
                static_cast<size_t>(row_base) * K;
            const float *tile_scales = d_scales_A +
                static_cast<size_t>(row_base) *
                    activation_blocks_per_row;
            float *tile_c = d_C +
                static_cast<size_t>(row_base) * N;
            const float *tile_existing = d_C_existing
                ? d_C_existing + static_cast<size_t>(row_base) * N
                : nullptr;

            nativeVnniGemvTensorCoreSmallM<CB><<<
                dim3((N + 31) / 32, (tile_m + 15) / 16, kb),
                128,
                0,
                stream>>>(
                tile_a,
                d_payload,
                d_scales,
                d_mins,
                d_emins,
                partials,
                tile_scales,
                tile_m,
                N,
                K,
                kb,
                alpha);
            if (cudaGetLastError() != cudaSuccess)
                return false;

            nativeVnniGemv_reduce_small_m<<<
                dim3((N + 255) / 256, tile_m),
                256,
                0,
                stream>>>(
                partials,
                tile_c,
                tile_existing,
                d_bias,
                tile_m,
                N,
                kb,
                beta);
            if (cudaGetLastError() != cudaSuccess)
                return false;
        }
        return true;
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

    template <int ROWS_PER_TILE, uint8_t CB, bool MATCH_WIDE_SERIAL = false>
    bool dispatchKparSmallMGeneratedRows(
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
            return launchKparSmallMImpl<32, 1, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 1:
            return launchKparSmallMImpl<64, 1, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 64 * 100 + 2:
            return launchKparSmallMImpl<64, 2, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 1:
            return launchKparSmallMImpl<128, 1, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 128 * 100 + 2:
            return launchKparSmallMImpl<128, 2, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 256 * 100 + 2:
            return launchKparSmallMImpl<256, 2, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
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
            return launchKparSmallMImpl<256, 4, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        case 512 * 100 + 4:
            return launchKparSmallMImpl<512, 4, CB, ROWS_PER_TILE, MATCH_WIDE_SERIAL>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                tuning.target_waves, tuning.mkg, tuning.max_kb,
                tuning.exact_kb,
                tuning.force_two_phase, cuda_device_id, stream, gemv_ctx);
        default:
            return false;
        }
    }

    /**
     * @brief Dispatch one trained grouped row-reuse instantiation.
     *
     * The row count only controls how many verifier rows share a decoded
     * weight block inside one CTA. Each row retains the generated public-M1
     * tile and exact K-partition schedule, so this parameter cannot alter that
     * row's reduction tree or publication bytes.
     */
    template <uint8_t CB, bool MATCH_WIDE_SERIAL = false>
    bool dispatchKparSmallMGenerated(
        const GeneratedDispatchTuning &tuning,
        int grouped_rows,
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
        CUDAGemvContext_ *gemv_ctx,
        int cuda_device_id, cudaStream_t stream)
    {
        switch (grouped_rows)
        {
        case 4:
            return dispatchKparSmallMGeneratedRows<4, CB, MATCH_WIDE_SERIAL>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, gemv_ctx, cuda_device_id, stream);
        case 8:
            return dispatchKparSmallMGeneratedRows<8, CB, MATCH_WIDE_SERIAL>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, gemv_ctx, cuda_device_id, stream);
        case 16:
            return dispatchKparSmallMGeneratedRows<16, CB, MATCH_WIDE_SERIAL>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, gemv_ctx, cuda_device_id, stream);
        case 32:
            return dispatchKparSmallMGeneratedRows<32, CB, MATCH_WIDE_SERIAL>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, gemv_ctx, cuda_device_id, stream);
        case 64:
            return dispatchKparSmallMGeneratedRows<64, CB, MATCH_WIDE_SERIAL>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, gemv_ctx, cuda_device_id, stream);
        case 2:
            return dispatchKparSmallMGeneratedRows<2, CB, MATCH_WIDE_SERIAL>(
                tuning, d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, M, N, K, alpha, beta,
                d_C_existing, d_bias, gemv_ctx, cuda_device_id, stream);
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

            // ROWPAR preparation is a setup transaction. The timed dispatch
            // path may consume an existing representation but cannot allocate
            // or transpose one.
            if (shape == NativeGemvShape::ROWPAR)
            {
                if (!rm_slot || !*rm_slot)
                    return false;
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
        if (!selectCachedGeneratedDispatch<CB>(
                kCanonicalDecodePolicyGraphCaptured,
                1,
                N,
                K,
                shape,
                tuning))
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
            gemv_ctx,
            0,
            kCanonicalDecodePolicyGraphCaptured ? 1 : 0);

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
        int grouped_rows = 0;
        bool grouped_tensor_core = false;

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
            const bool dispatch_policy_graph_captured =
                decode_equivalent_m1
                    ? kCanonicalDecodePolicyGraphCaptured
                    : graph_captured;
            if (!selectCachedGeneratedDispatch<CB>(
                    dispatch_policy_graph_captured,
                    dispatch_m,
                    N,
                    K,
                    shape,
                    tuning))
            {
                return false;
            }
            if (decode_equivalent_m1)
            {
                /*
                 * Row reuse is a separately learned economy axis. It cannot
                 * alter the public-M1 family, output tile, or exact K
                 * partition selected above. A production artifact without a
                 * total grouped policy is incomplete and must remain a hard
                 * miss instead of silently reverting to the historical
                 * two-row schedule.
                 */
                if (g_grouped_tensor_core_override)
                {
                    grouped_tensor_core = true;
                }
                else if (g_grouped_rows_override > 0)
                {
                    /* Trainer-only candidate forcing; M1 arithmetic remains generated. */
                    grouped_rows = g_grouped_rows_override;
                }
                else
                {
#if defined(LLAMINAR_CUDA_GROUPED_DISPATCH_POLICY_V2)
                    GeneratedGroupedTuning grouped_tuning{};
                    if (!selectCachedGeneratedGroupedTuning<CB>(
                            dispatch_policy_graph_captured,
                            M,
                            N,
                            K,
                            grouped_tuning))
                        return false;
                    switch (grouped_tuning.kernel)
                    {
                    case GeneratedGroupedKernel::Dp4aRows:
                        grouped_rows = grouped_tuning.grouped_rows;
                        if (grouped_rows <= 0)
                            return false;
                        break;
                    case GeneratedGroupedKernel::TensorCoreMma16:
                        grouped_tensor_core = true;
                        break;
                    default:
                        return false;
                    }
#else
                    return false;
#endif
                }
            }
        }

        const bool decode_equivalent_m1 = decodeEquivalentM1ConfigActive();
        if (grouped_tensor_core)
        {
            if (!decode_equivalent_m1 || g_sweep.active)
                return false;
            const bool launched = launchTensorCoreSmallMGenerated<CB>(
                shape,
                tuning,
                d_A_int8,
                d_payload,
                d_scales,
                d_mins,
                d_emins,
                d_C,
                d_scales_A,
                M,
                N,
                K,
                alpha,
                beta,
                d_C_existing,
                d_bias,
                gemv_ctx,
                stream);
            if (!launched)
                return false;
            recordTensorCoreVerifierDispatch<CB>(
                tuning,
                M,
                N,
                K,
                g_last_effective_kb,
                cuda_device_id,
                graph_captured);
            return true;
        }
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
            gemv_ctx,
            grouped_rows,
            decode_equivalent_m1
                ? (kCanonicalDecodePolicyGraphCaptured ? 1 : 0)
                : (graph_captured ? 1 : 0));
        if (shape == NativeGemvShape::WIDE || shape == NativeGemvShape::DIRECT)
        {
            if (!decode_equivalent_m1)
                return false;
            return dispatchKparSmallMGenerated<CB, true>(
                tuning, grouped_rows,
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, M, N, K, alpha, beta, d_C_existing, d_bias,
                gemv_ctx, cuda_device_id, stream);
        }

        if constexpr (CB != 19)
        {
            /* Explicit diagnostic ROWPAR candidates require setup preparation. */
            if (shape == NativeGemvShape::ROWPAR && rm_slot)
            {
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
                tuning, grouped_rows,
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
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

/**
 * @brief Measure the in-TU cached selector used by a production specialization.
 *
 * The public single-query test shim necessarily adds an out-of-line function
 * call, a runtime codebook switch, and four result stores that production's
 * templated launcher does not pay. Keep the timing loop beside the selector so
 * the perf regression measures the inlined path. The loop-only control carries
 * identical query indexing and compiler fences and is subtracted from the
 * selector interval.
 */
template <uint8_t CB>
static double measureCachedGeneratedDispatchNs(
    bool graph_captured,
    int m,
    const int *ns,
    const int *ks,
    int query_count,
    int iterations,
    uint64_t *out_checksum)
{
    if (!ns || !ks || query_count <= 0 || iterations <= 0)
        return -1.0;

    uint64_t checksum = 0;
    const auto baseline_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const int index = iteration % query_count;
        checksum += static_cast<uint64_t>(ns[index] + ks[index]);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    const auto baseline_end = std::chrono::steady_clock::now();

    const auto selector_begin = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration)
    {
        const int index = iteration % query_count;
        NativeGemvShape shape = NativeGemvShape::KPAR;
        GeneratedDispatchTuning tuning{};
        const bool selected = selectCachedGeneratedDispatch<CB>(
            graph_captured,
            m,
            ns[index],
            ks[index],
            shape,
            tuning);
        checksum += static_cast<uint64_t>(selected) |
                    (static_cast<uint64_t>(static_cast<int>(shape) + 1) << 1) |
                    (static_cast<uint64_t>(tuning.tile_n) << 8) |
                    (static_cast<uint64_t>(tuning.cpt) << 20) |
                    (static_cast<uint64_t>(tuning.exact_kb) << 24);
        std::atomic_signal_fence(std::memory_order_seq_cst);
    }
    const auto selector_end = std::chrono::steady_clock::now();

    if (out_checksum)
        *out_checksum = checksum;
    const double baseline_ns = std::chrono::duration<double, std::nano>(
        baseline_end - baseline_begin).count();
    const double selector_ns = std::chrono::duration<double, std::nano>(
        selector_end - selector_begin).count();
    return std::max(
        0.0,
        (selector_ns - baseline_ns) / static_cast<double>(iterations));
}

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

    /**
     * @brief Query cached generated decode policy without touching a device.
     *
     * This host-only surface exists for dispatch latency and totality tests.
     * Production launchers call the same cached template directly.
     */
    bool cudaNativeVNNIGemvTuned_queryGeneratedDispatch(
        uint8_t codebook_id,
        int graph_captured,
        int m,
        int n,
        int k,
        int *shape_id,
        int *tile_n,
        int *cpt,
        int *exact_kb)
    {
        NativeGemvShape shape = NativeGemvShape::KPAR;
        GeneratedDispatchTuning tuning{};
        if (!queryCachedGeneratedDispatch(
                codebook_id,
                graph_captured != 0,
                m,
                n,
                k,
                shape,
                tuning))
        {
            return false;
        }
        if (shape_id)
            *shape_id = static_cast<int>(shape);
        if (tile_n)
            *tile_n = tuning.tile_n;
        if (cpt)
            *cpt = tuning.cpt;
        if (exact_kb)
            *exact_kb = tuning.exact_kb;
        return true;
    }

    /**
     * @brief Resolve the immutable production M=1 reduction schedule.
     *
     * Grouped tensor-core GEMM owns several rows economically, but each row
     * must reproduce the exact public serial-decode expression tree. The
     * generated M=1 policy is the authority for whether publication is a
     * single accumulator or an ordered reduction of K-partition partials.
     * This host query exposes only that arithmetic contract; it does not
     * launch work, allocate memory, or permit the prefill dispatcher to invent
     * a replacement policy.
     *
     * @param codebook_id NativeVNNI codebook identifier.
     * @param n Output-column count.
     * @param k Reduction-column count.
     * @param sm_count Physical SM count used by generated KPAR resolution.
     * @param uses_ordered_reducer Receives one for KPAR and zero for a direct
     *        single-accumulator route.
     * @param k_partitions Receives the exact number of serial K partitions.
     * @return true only when the complete production schedule is available.
     */
    bool cudaNativeVNNIGemvTuned_queryCanonicalM1Schedule(
        uint8_t codebook_id,
        int n,
        int k,
        int sm_count,
        int *uses_ordered_reducer,
        int *k_partitions)
    {
        if (n <= 0 || k <= 0 || (k % BLOCK_K) != 0 || sm_count <= 0 ||
            !uses_ordered_reducer || !k_partitions)
        {
            return false;
        }

        NativeGemvShape shape = NativeGemvShape::KPAR;
        GeneratedDispatchTuning tuning{};
        if (!queryCachedGeneratedDispatch(
                codebook_id,
                kCanonicalDecodePolicyGraphCaptured,
                1,
                n,
                k,
                shape,
                tuning))
        {
            return false;
        }

        switch (shape)
        {
        case NativeGemvShape::WIDE:
        case NativeGemvShape::DIRECT:
            *uses_ordered_reducer = 0;
            *k_partitions = 1;
            return true;
        case NativeGemvShape::KPAR:
        {
            const int resolved_partitions = resolveKparKBlocks(
                (n + tuning.tile_n - 1) / tuning.tile_n,
                k / BLOCK_K,
                sm_count,
                tuning.target_waves,
                tuning.mkg,
                tuning.max_kb,
                tuning.exact_kb);
            if (resolved_partitions <= 0)
                return false;
            *uses_ordered_reducer = 1;
            *k_partitions = resolved_partitions;
            return true;
        }
        case NativeGemvShape::ROWPAR:
            /*
             * ROWPAR has a warp-shuffle tree rather than an ordered K-partial
             * reducer. It is not currently emitted by the production policy;
             * fail hard if a future policy introduces it without teaching
             * grouped tensor-core publication the corresponding tree.
             */
            return false;
        }
        return false;
    }

    /**
     * @brief Host-only perf surface for the inlined generated selector.
     *
     * No CUDA API, allocation, transfer, synchronization, or kernel launch is
     * performed. This function exists solely for the offline dispatch-latency
     * regression.
     */
    double cudaNativeVNNIGemvTuned_measureGeneratedDispatchNs(
        uint8_t codebook_id,
        int graph_captured,
        int m,
        const int *ns,
        const int *ks,
        int query_count,
        int iterations,
        uint64_t *out_checksum)
    {
#define LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(CODEBOOK) \
        case CODEBOOK: \
            return measureCachedGeneratedDispatchNs<CODEBOOK>( \
                graph_captured != 0, m, ns, ks, query_count, iterations, \
                out_checksum)
        switch (codebook_id)
        {
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(0);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(4);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(5);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(6);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(7);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(8);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(9);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(10);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(11);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(12);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(13);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(14);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(15);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(16);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(17);
            LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK(19);
        default:
            return -1.0;
        }
#undef LLAMINAR_MEASURE_CUDA_NVNNI_CODEBOOK
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
        g_grouped_rows_override = 0;
        g_grouped_tensor_core_override = false;
    }

    /**
     * @brief Select one trainer-only grouped row-reuse instantiation.
     * @param rows_per_tile Exact compile-time row tile to benchmark.
     *
     * This does not activate the ordinary M1 sweep override: grouped verifier
     * calls must continue to inherit the generated public-M1 family, tile, and
     * K-partition schedule while varying only weight reuse across rows.
     */
    void cudaNativeVNNIGemvSweep_setGroupedRows(int rows_per_tile)
    {
        g_grouped_rows_override = rows_per_tile;
    }

    /**
     * @brief Select the integer tensor-core grouped verifier candidate.
     * @param enabled Non-zero only inside the trainer's candidate scope.
     *
     * The low-level dispatcher additionally requires decode-equivalent verifier
     * mode and rejects an ordinary M1 sweep override. This function therefore
     * cannot silently alter the production route if a diagnostic scope leaks.
     */
    void cudaNativeVNNIGroupedVerifier_setTensorCoreOverride(int enabled)
    {
        g_grouped_tensor_core_override = enabled != 0;
    }

    bool cudaNativeVNNIGemvSweep_isActive()
    {
        return g_sweep.active;
    }

    void cudaNativeVNNIGemvTuned_clearStaticState()
    {
        // Row-major weight data is now owned by CUDAQuantisedGemmKernel::Impl
        // (per-weight) and KPAR partials are owned by CUDAGemvContext
        // (per-device).
        // Only sweep override needs clearing here.
        g_sweep.active = false;
        g_grouped_rows_override = 0;
        g_grouped_tensor_core_override = false;
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
                transposeBuffer<uint8_t, 6>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 8:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 8>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 9:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 9>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 12:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 12>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 13:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 13>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 2:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 2>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 4:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 4>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 16:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 16>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 20:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 20>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 24:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 24>(d_payload_col, N, K_blocks, device_id, s));
            break;
        case 32:
            rm->d_payload = reinterpret_cast<uint8_t *>(
                transposeBuffer<uint8_t, 32>(d_payload_col, N, K_blocks, device_id, s));
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
        rm->d_scales = transposeBuffer<uint16_t, 2>(
            d_scales_col, N, K_blocks, device_id, s);
        if (!rm->d_scales)
        {
            llaminar2::getCUDABackend()->free(rm->d_payload, device_id);
            delete rm;
            return nullptr;
        }

        // Transpose mins if present
        if (d_mins_col)
        {
            rm->d_mins = transposeBuffer<uint16_t, 2>(
                d_mins_col, N, K_blocks, device_id, s);
            if (!rm->d_mins)
            {
                auto *const backend = llaminar2::getCUDABackend();
                backend->free(rm->d_payload, device_id);
                backend->free(rm->d_scales, device_id);
                delete rm;
                return nullptr;
            }
        }

        // Transpose emins if present
        if (d_emins_col)
        {
            rm->d_emins = transposeBuffer<uint32_t, 4>(
                d_emins_col, N, K_blocks, device_id, s);
            if (!rm->d_emins)
            {
                auto *const backend = llaminar2::getCUDABackend();
                backend->free(rm->d_payload, device_id);
                backend->free(rm->d_scales, device_id);
                if (rm->d_mins)
                    backend->free(rm->d_mins, device_id);
                delete rm;
                return nullptr;
            }
        }

        cudaStreamSynchronize(s);
        return rm;
    }

    CUDARowMajorWeights *cudaRowMajorWeights_createForCodebook(
        const uint8_t *d_payload_col,
        const uint16_t *d_scales_col,
        const uint16_t *d_mins_col,
        const uint32_t *d_emins_col,
        int N,
        int K,
        uint8_t codebook_id,
        int device_id,
        void *stream)
    {
#define LLAMINAR_CREATE_CUDA_ROW_MAJOR(CODEBOOK)                         \
        case CODEBOOK:                                                   \
            return cudaRowMajorWeights_create(                           \
                d_payload_col, d_scales_col, d_mins_col, d_emins_col,    \
                N, K,                                                    \
                llaminar2::cuda_native_vnni::                            \
                    CodebookTraits<CODEBOOK>::payload_bytes,              \
                device_id, stream)
        switch (codebook_id)
        {
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(0);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(4);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(5);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(6);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(7);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(8);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(9);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(10);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(11);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(12);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(13);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(14);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(15);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(16);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(17);
            LLAMINAR_CREATE_CUDA_ROW_MAJOR(19);
        default:
            return nullptr;
        }
#undef LLAMINAR_CREATE_CUDA_ROW_MAJOR
    }

    bool cudaNativeVNNIGemvTuned_policyRequiresRowMajor(
        uint8_t codebook_id,
        int N,
        int K)
    {
        for (const bool graph_captured : {false, true})
        {
            NativeGemvShape shape = NativeGemvShape::KPAR;
            GeneratedDispatchTuning tuning{};
            if (queryCachedGeneratedDispatch(
                    codebook_id,
                    graph_captured,
                    1,
                    N,
                    K,
                    shape,
                    tuning) &&
                shape == NativeGemvShape::ROWPAR)
            {
                return true;
            }
        }
        return false;
    }

    void cudaRowMajorWeights_destroy(CUDARowMajorWeights *rm)
    {
        if (!rm)
            return;
        cudaSetDevice(rm->device_id);
        auto *const backend = llaminar2::getCUDABackend();
        if (!backend)
        {
            /*
             * Releasing only the host handle would leak four device allocations
             * and conceal a broken backend lifetime.  This owner is destroyed
             * before backend teardown by construction; violating that order is
             * a fatal lifecycle defect, not a recoverable cleanup condition.
             */
            std::terminate();
        }
        if (rm->d_payload)
            backend->free(rm->d_payload, rm->device_id);
        if (rm->d_scales)
            backend->free(rm->d_scales, rm->device_id);
        if (rm->d_mins)
            backend->free(rm->d_mins, rm->device_id);
        if (rm->d_emins)
            backend->free(rm->d_emins, rm->device_id);
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

extern "C" void cudaNativeVNNIGemvTuned_setSerialPartitionN(int n)
{
    g_cuda_native_vnni_serial_partition_n = n;
}

extern "C" int cudaNativeVNNIGemvTuned_getSerialPartitionN()
{
    return g_cuda_native_vnni_serial_partition_n;
}
