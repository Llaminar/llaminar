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

#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <mutex>

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
    // SM count query (cached)
    // =====================================================================
    static int getSmCount()
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

    // =====================================================================
    // Row-major weight layout cache for row-parallel GEMV
    //
    // Column-major layout (current) is coalesced for column-parallel KPAR,
    // but row-parallel (grid=N, one block per output row) needs row-major
    // for coalesced K-dimension reads.  This cache lazily transposes weights
    // on first use and reuses the transposed copy for subsequent GEMV calls.
    //
    // Memory cost: ~1× the weight data (about 1.4 GB for 7B Q4_0).
    // Controlled by LLAMINAR_CUDA_GEMV_ROWPAR env variable.
    // =====================================================================
    struct RowMajorEntry
    {
        uint8_t *d_payload = nullptr;
        uint16_t *d_scales = nullptr;
        uint16_t *d_mins = nullptr;
        uint32_t *d_emins = nullptr;
        int N = 0;
        int K_blocks = 0;
        int device_id = -1;
    };

    static std::unordered_map<const void *, RowMajorEntry> g_rm_cache;
    static std::mutex g_rm_mutex;

    // GPU kernel: transpose blocks from column-major to row-major
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
            const char *env = std::getenv("LLAMINAR_CUDA_GEMV_ROWPAR");
            enabled = (env && env[0] == '0') ? 0 : 1;
        }
        return enabled == 1;
    }

    // Get or create row-major weight cache for a given column-major payload.
    // Returns the cached entry, or nullptr fields if creation fails.
    // Must be called from the CUDA device context.
    static const RowMajorEntry *getOrCreateRowMajor(
        const uint8_t *d_payload_col,
        const uint16_t *d_scales_col,
        const uint16_t *d_mins_col,
        const uint32_t *d_emins_col,
        int N, int K, int payload_bytes,
        int device_id, cudaStream_t stream)
    {
        std::lock_guard<std::mutex> lock(g_rm_mutex);

        auto it = g_rm_cache.find(d_payload_col);
        if (it != g_rm_cache.end())
            return &it->second;

        const int K_blocks = K / BLOCK_K;

        // Check free memory before allocating.  The RM cache for large models
        // (Q8_0 @ 7B = ~6.7 GB) can exhaust VRAM and leave no headroom for
        // CUDA graph instantiation.  Keep ≥1.5 GB free.
        {
            size_t free_bytes = 0, total_bytes_gpu = 0;
            if (cudaMemGetInfo(&free_bytes, &total_bytes_gpu) == cudaSuccess)
            {
                const size_t need = static_cast<size_t>(N) * K_blocks *
                                    (payload_bytes + 2 /* scales */);
                constexpr size_t HEADROOM = size_t{1536} << 20; // 1.5 GB
                if (free_bytes < need + HEADROOM)
                    return nullptr;
            }
        }

        RowMajorEntry entry;
        entry.N = N;
        entry.K_blocks = K_blocks;
        entry.device_id = device_id;

        // Transpose payload (Q8_0 uses column-major ROWPAR, so no case 32 needed)
        switch (payload_bytes)
        {
        case 16:
            entry.d_payload = transposeBuffer<uint8_t, 16>(d_payload_col, N, K_blocks, stream);
            break;
        case 12:
            entry.d_payload = transposeBuffer<uint8_t, 12>(d_payload_col, N, K_blocks, stream);
            break;
        case 8:
            entry.d_payload = transposeBuffer<uint8_t, 8>(d_payload_col, N, K_blocks, stream);
            break;
        case 6:
            entry.d_payload = transposeBuffer<uint8_t, 6>(d_payload_col, N, K_blocks, stream);
            break;
        case 4:
            entry.d_payload = transposeBuffer<uint8_t, 4>(d_payload_col, N, K_blocks, stream);
            break;
        case 2:
            entry.d_payload = transposeBuffer<uint8_t, 2>(d_payload_col, N, K_blocks, stream);
            break;
        default:
            return nullptr;
        }
        if (!entry.d_payload)
            return nullptr;

        // Transpose scales (always 2 bytes per block)
        entry.d_scales = transposeBuffer<uint16_t, 2>(d_scales_col, N, K_blocks, stream);
        if (!entry.d_scales)
        {
            cudaFree(entry.d_payload);
            return nullptr;
        }

        // Transpose mins if present
        if (d_mins_col)
        {
            entry.d_mins = transposeBuffer<uint16_t, 2>(d_mins_col, N, K_blocks, stream);
            if (!entry.d_mins)
            {
                cudaFree(entry.d_payload);
                cudaFree(entry.d_scales);
                return nullptr;
            }
        }

        // Transpose emins if present
        if (d_emins_col)
        {
            entry.d_emins = transposeBuffer<uint32_t, 4>(d_emins_col, N, K_blocks, stream);
            if (!entry.d_emins)
            {
                cudaFree(entry.d_payload);
                cudaFree(entry.d_scales);
                if (entry.d_mins)
                    cudaFree(entry.d_mins);
                return nullptr;
            }
        }

        // Sync to ensure transpose is complete before first use
        cudaStreamSynchronize(stream);

        auto [ins_it, ok] = g_rm_cache.emplace(d_payload_col, entry);
        return ok ? &ins_it->second : nullptr;
    }

    // =====================================================================
    // Workspace cache for two-phase KPAR reduction
    //
    // Lazily allocates a device buffer for partial sums. Grows as needed.
    // Max size is modest: e.g. 14B_FFN_Down N=5120 × kb=40 = 800KB.
    // =====================================================================
    static float *getKparPartials(size_t num_floats, int device_id)
    {
        static float *s_ptr = nullptr;
        static size_t s_capacity = 0;
        static int s_device = -1;

        if (s_ptr && s_capacity >= num_floats && s_device == device_id)
            return s_ptr;

        if (s_ptr)
        {
            cudaSetDevice(s_device);
            cudaFree(s_ptr);
            s_ptr = nullptr;
        }

        cudaSetDevice(device_id);
        if (cudaMalloc(&s_ptr, num_floats * sizeof(float)) != cudaSuccess)
        {
            s_ptr = nullptr;
            s_capacity = 0;
            return nullptr;
        }
        s_capacity = num_floats;
        s_device = device_id;
        return s_ptr;
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

        // Prefer factor of k_groups to avoid uneven splits
        if (k_groups % kb != 0)
        {
            for (int d = 1; d < kb; ++d)
            {
                if (kb - d >= 2 && k_groups % (kb - d) == 0)
                {
                    kb -= d;
                    break;
                }
                if (kb + d <= kb_max && k_groups % (kb + d) == 0)
                {
                    kb += d;
                    break;
                }
            }
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
            int32_t a_vals[8];
            {
                const int32_t *a_ptr = reinterpret_cast<const int32_t *>(
                    d_A_int8 + blk * BLOCK_K);
#pragma unroll
                for (int g = 0; g < 8; ++g)
                    a_vals[g] = a_ptr[g];
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
#pragma unroll
            for (int g = 0; g < 8; ++g)
                a_vals[g] = *reinterpret_cast<const int32_t *>(
                    d_A_int8 + blk * BLOCK_K + g * 4);

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
        int device_id, cudaStream_t stream)
    {
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = getSmCount();
        const int kb = selectKSplit(grid_n, k_groups, num_sms,
                                    target_waves, min_kgroups_per_cta);
        const int kb_capped = (max_kb > 0) ? std::min(kb, max_kb) : kb;

        // Choose two-phase vs atomic based on partials buffer size
        const size_t partials_bytes = static_cast<size_t>(kb_capped) * N * sizeof(float);
        const bool use_two_phase = (partials_bytes <= kTwoPhaseMaxBytes);

        dim3 grid(grid_n, kb_capped);

        if (use_two_phase)
        {
            // Two-phase: write partials, then reduce
            float *d_partials = getKparPartials(
                static_cast<size_t>(kb_capped) * N, device_id);
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

    template <uint8_t CB>
    bool launchKpar(
        const int8_t *d_A_int8, const uint8_t *d_payload,
        const uint16_t *d_scales, const uint16_t *d_mins,
        const uint32_t *d_emins, float *d_C,
        const float *d_scales_A, int N, int K,
        float alpha, float beta,
        const float *d_C_existing, const float *d_bias,
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
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream);
        case KparTile::T64_C1:
            return launchKparImpl<64, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream);
        case KparTile::T64_C2:
            return launchKparImpl<64, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream);
        case KparTile::T128_C1:
            return launchKparImpl<128, 1, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream);
        case KparTile::T128_C2:
            return launchKparImpl<128, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream);
        case KparTile::T256_C2:
            return launchKparImpl<256, 2, CB>(
                d_A_int8, d_payload, d_scales, d_mins, d_emins,
                d_C, d_scales_A, N, K, alpha, beta,
                d_C_existing, d_bias, tw, mkg, mkb, cuda_device_id, stream);
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
                    tuning.force_two_phase, cuda_device_id, stream);
            case 64 * 100 + 1:
                return sweepLaunchKpar<64, 1, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, cuda_device_id, stream);
            case 64 * 100 + 2:
                return sweepLaunchKpar<64, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, cuda_device_id, stream);
            case 128 * 100 + 1:
                return sweepLaunchKpar<128, 1, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, cuda_device_id, stream);
            case 128 * 100 + 2:
                return sweepLaunchKpar<128, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, cuda_device_id, stream);
            case 256 * 100 + 2:
                return sweepLaunchKpar<256, 2, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, cuda_device_id, stream);
            case 256 * 100 + 4:
                return sweepLaunchKpar<256, 4, CB>(
                    d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                    d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                    tuning.target_waves, tuning.mkg, tuning.max_kb,
                    tuning.force_two_phase, cuda_device_id, stream);
            default:
                return false;
            }
        }

        if (shape == NativeGemvShape::ROWPAR)
        {
            // Look up the row-major cache using the column-major d_payload as key
            const RowMajorEntry *rm = nullptr;
            {
                std::lock_guard<std::mutex> lock(g_rm_mutex);
                auto it = g_rm_cache.find(d_payload);
                if (it != g_rm_cache.end())
                    rm = &it->second;
            }
            if (!rm || !rm->d_payload || !rm->d_scales)
                return false;

            // tile_n encodes NWARPS for ROWPAR (2, 4, 8)
            return launchRowPar<CB>(
                d_A_int8, rm->d_payload, rm->d_scales, rm->d_mins, rm->d_emins, d_C,
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
        int device_id, cudaStream_t stream);

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

            // For ROWPAR sweep: ensure row-major cache exists (skip for Q8_0
            // which uses column-major ROWPAR to avoid doubling VRAM)
            if (shape == NativeGemvShape::ROWPAR)
            {
                if constexpr (CB != 19)
                {
                    constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
                    const auto *rm = getOrCreateRowMajor(
                        d_payload, d_scales, d_mins, d_emins,
                        N, K, PB, cuda_device_id, stream);
                    if (!rm || !rm->d_payload)
                        return false;
                }
            }

            return dispatchGeneratedTuning<CB>(
                shape, tuning,
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
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
                constexpr int PB = llaminar2::cuda_native_vnni::CodebookTraits<CB>::payload_bytes;
                const auto *rm = getOrCreateRowMajor(
                    d_payload, d_scales, d_mins, d_emins,
                    N, K, PB, cuda_device_id, stream);
                if (rm && rm->d_payload)
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

        return dispatchGeneratedTuning<CB>(
            shape, tuning,
            d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
            d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
            cuda_device_id, stream);
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
        int device_id, cudaStream_t stream)
    {
        constexpr int THREADS = TILE_N / CPT;

        const int grid_n = (N + TILE_N - 1) / TILE_N;
        const int k_groups = K / BLOCK_K;
        const int num_sms = getSmCount();
        const int kb = selectKSplit(grid_n, k_groups, num_sms,
                                    target_waves, min_kgroups_per_cta);
        const int kb_capped = (max_kb > 0) ? std::min(kb, max_kb) : kb;

        const size_t partials_bytes = static_cast<size_t>(kb_capped) * N * sizeof(float);
        bool use_two_phase;
        if (force_two_phase == 1)
            use_two_phase = true;
        else if (force_two_phase == 2)
            use_two_phase = false;
        else
            use_two_phase = (partials_bytes <= kTwoPhaseMaxBytes);

        dim3 grid(grid_n, kb_capped);

        if (use_two_phase)
        {
            float *d_partials = getKparPartials(
                static_cast<size_t>(kb_capped) * N, device_id);
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
        void *stream)
    {
        if (!d_A_int8 || !d_payload || !d_scales || !d_C_fp32 || !d_scales_A_block)
            return false;
        if (N <= 0 || K <= 0 || (K % BLOCK_K) != 0)
            return false;
        if (!cudaNativeVNNIGemvTuned_supportsCodebook(codebook_id))
            return false;

        cudaError_t err = cudaSetDevice(cuda_device_id);
        if (err != cudaSuccess)
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

        switch (codebook_id)
        {
        case 0:
            return dispatchCodebook<0>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 4:
            return dispatchCodebook<4>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 5:
            return dispatchCodebook<5>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 6:
            return dispatchCodebook<6>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 7:
            return dispatchCodebook<7>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 8:
            return dispatchCodebook<8>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 9:
            return dispatchCodebook<9>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 10:
            return dispatchCodebook<10>(d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 11:
            return dispatchCodebook<11>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 12:
            return dispatchCodebook<12>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 13:
            return dispatchCodebook<13>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 14:
            return dispatchCodebook<14>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 15:
            return dispatchCodebook<15>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 16:
            return dispatchCodebook<16>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 17:
            return dispatchCodebook<17>(d_A_int8, d_payload, d_scales, d_mins, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        case 19:
            return dispatchCodebook<19>(d_A_int8, d_payload, d_scales, nullptr, nullptr, d_C_fp32, d_scales_A_block, N, K, alpha, beta, d_C_existing, d_bias, cuda_device_id, cuda_stream);
        default:
            return false;
        }
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
}
