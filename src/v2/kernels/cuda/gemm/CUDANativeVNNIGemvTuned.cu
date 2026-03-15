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
        DIRECT
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

        __shared__ int32_t smem_A[8];

        float acc[CPT];
#pragma unroll
        for (int c = 0; c < CPT; ++c)
            acc[c] = 0.0f;

        for (int blk = blk_begin; blk < blk_end; ++blk)
        {
            if (threadIdx.x < 8)
            {
                smem_A[threadIdx.x] = *reinterpret_cast<const int32_t *>(
                    d_A_int8 + blk * BLOCK_K + threadIdx.x * 4);
            }
            __syncthreads();

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
                if constexpr (TWO_PHASE)
                    llaminar2::cuda_native_vnni::decode_groups_vec<CB>(payload, packed_groups);
                else
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

            __syncthreads();
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
            return dispatchGeneratedTuning<CB>(
                shape, tuning,
                d_A_int8, d_payload, d_scales, d_mins, d_emins, d_C,
                d_scales_A, N, K, alpha, beta, d_C_existing, d_bias,
                cuda_device_id, stream);
        }

        // Normal dispatch path
        const NativeGemvShape shape = classifyShapeGenerated<CB>(N, K);
        const GeneratedDispatchTuning tuning = selectGeneratedTuning<CB>(N, K);
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
