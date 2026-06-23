/**
 * @file ROCmGemvKernel_INT8_VNNI.hip
 * @brief INT8 VNNI GEMV/GEMM kernels and dispatch for AMD MI50/MI60 (gfx906)
 *
 * This file contains all INT8-based VNNI kernels for decode (M=1) GEMV and
 * prefill (M>1) GEMM, including:
 *   - Basic INT8 VNNI decode kernels (1-thread-per-column, wide, square, grid-kpar)
 *   - LDS K-reduction decode kernels
 *   - Fused FP32→INT8 decode kernels (quantize + GEMV + scale in one launch)
 *   - Scatter/reduce and self-reducing scatter pipelines
 *   - Batched scatter dispatch (QKV/GateUp in one launch)
 *   - INT8-input direct and scatter kernels (pre-quantized activations)
 *   - INT8 VNNI prefill GEMM kernels with tiling variants
 *   - VNNI→row-major weight repack utility
 *
 * Split from ROCmGemvKernel.hip. Common constants and structs are defined
 * in ROCmGemvKernel.hip (the main file).
 *
 * @see ROCmGemvKernel.hip for shared constants and GemvBatchArgs
 * @see ROCmGemvKernel_native_VNNI.hip for native-VNNI 4-bit kernels
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <atomic>

#include "../../../backends/rocm/HipDeviceGuard.h"

// Tile width: number of output columns per workgroup
// 256 threads per block — each thread does one output column
constexpr int GEMV_TILE_N = 256;

// VNNI kernel tile parameters
constexpr int GEMV_INT8_VNNI_WIDE_TILE_N = 128;
constexpr int GEMV_INT8_VNNI_WIDE_DEFAULT_CPT = 2;
constexpr int GEMV_INT8_VNNI_WIDE2_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_SQUARE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_KPAR_TILE_K = 256;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_TILE_N = 128;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_CPT = 2;

constexpr int MAX_GEMV_BATCH = 8;

struct GemvBatchArgs
{
    const int8_t* d_B[MAX_GEMV_BATCH];         // VNNI weights per projection
    float*        d_C[MAX_GEMV_BATCH];         // FP32 output per projection
    const float*  d_scales_B[MAX_GEMV_BATCH];  // Weight scales per projection [N_i]
    const float*  d_bias[MAX_GEMV_BATCH];      // Bias per projection (nullable)
    int           N[MAX_GEMV_BATCH];           // Output dim per projection
    int           grid_n_offset[MAX_GEMV_BATCH]; // Exclusive prefix sum of grid_n
    int           n_offset[MAX_GEMV_BATCH];      // Exclusive prefix sum of N
    int           num_batches;
    int           total_grid_n;                // sum of all grid_n
    int           total_N;                     // sum of all N
};

static std::atomic<int> g_int8_vnni_tn_override{-1};
static std::atomic<int> g_int8_vnni_kb_override{-1};
static std::atomic<int> g_int8_vnni_wk_override{-1};
static std::atomic<int> g_int8_vnni_unroll_override{-1};  // 1=standard, 2=2x act_block unroll
static std::atomic<int> g_int8_vnni_act_block_override{-1};  // -1=default(32), 32, 64, 128
static std::atomic<int> g_int8_vnni_wide_tn_override{-1};
static std::atomic<int> g_int8_vnni_wide_cpt_override{-1};
static std::atomic<int> g_int8_vnni_wide_vec_load_override{-1};
static std::atomic<int> g_int8_vnni_skip_memset_override{0};  // 1 = skip hipMemsetAsync (for benchmarking)
static std::atomic<int> g_int8_vnni_force_pair_override{0};  // 1 = force batched pair path (skip split-to-singles)

static inline int select_blockwise_split_count(int grid_n, int act_blocks, int kb_override)
{
    constexpr int NUM_CUS = 60;
    constexpr int MIN_ACT_BLOCKS_PER_WAVE = 2;

    const int target_waves_per_cu = (grid_n <= 4) ? 2 : ((grid_n <= 8) ? 4 : 8);
    int target_blocks = target_waves_per_cu * NUM_CUS;
    int kb_raw = max(1, (target_blocks + grid_n - 1) / grid_n);
    int kb_max = max(1, act_blocks / MIN_ACT_BLOCKS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (act_blocks % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && act_blocks % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && act_blocks % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb_override > 0)
    {
        kb = min(kb_override, act_blocks);
    }

    return max(1, kb);
}

static inline int select_blockwise_lds_kreduce_outer_splits(int grid_n, int act_blocks, int kb_override)
{
    constexpr int NUM_CUS = 60;
    constexpr int WK = 4;
    constexpr int MIN_ACT_BLOCKS_PER_WAVE = 2;

    const int target_waves = 8 * NUM_CUS;
    int kb_raw = max(1, (target_waves + (grid_n * WK) - 1) / (grid_n * WK));
    int kb_max = max(1, act_blocks / (MIN_ACT_BLOCKS_PER_WAVE * WK));
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (act_blocks % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && act_blocks % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && act_blocks % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb_override > 0)
    {
        kb = min(kb_override, act_blocks);
    }

    return max(1, kb);
}

static inline int select_blockwise_tiny_kv_outer_splits(int grid_n, int act_blocks, int kb_override)
{
    constexpr int NUM_CUS = 60;
    constexpr int WK = 8;
    constexpr int MIN_ACT_BLOCKS_PER_WAVE = 1;

    const int target_waves = 8 * NUM_CUS;
    int kb_raw = max(1, (target_waves + (grid_n * WK) - 1) / (grid_n * WK));
    int kb_max = max(1, act_blocks / (MIN_ACT_BLOCKS_PER_WAVE * WK));
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (act_blocks % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && act_blocks % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && act_blocks % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb_override > 0)
    {
        kb = min(kb_override, act_blocks);
    }

    return max(1, kb);
}

static inline int select_blockwise_qwo_outer_splits(int grid_n, int act_blocks, int kb_override, int wk = 8)
{
    if (kb_override > 0)
    {
        return max(1, min(kb_override, act_blocks));
    }

    // Two regimes based on K-dimension work per wave:
    //
    // Small K (Q/Wo: act_blocks=112, acts_per_wave_single=14):
    //   Each wave has little work at KB=1. Maximize CU parallelism by splitting
    //   to ~1 act_block per wave → KB = act_blocks/wk (= 14 for Q/Wo).
    //   Exception: when grid_n alone saturates bandwidth (grid_n >= 2×NUM_CUS),
    //   additional KB splits only add atomicAdd overhead without benefit.
    //   Verified by sweep: FFN Gate/Up TP=1 at grid_n=148 → KB=1 is 3.7% faster.
    //
    // Large K (FFN Down: act_blocks=296-592, acts_per_wave_single=37-74):
    //   Three-branch approach based on CU occupancy at KB=1:
    //   - High grid_n (≥3 blocks/CU): K-walk is the bottleneck, not occupancy.
    //     Target 5 acts/shard to reduce critical path.
    //     Verified: 32B FFN Gate TP=1 grid_n=216 → KB=4 is 12.5% faster than KB=1.
    //   - Moderate grid_n (≥1.5 blocks/CU): conservative target of 16 acts/shard
    //     avoids over-splitting where KB=1 is near-optimal.
    //     Verified: 14B FFN Gate TP=1 grid_n=108 → KB=1 is optimal.
    //   - Low grid_n (<1.5 blocks/CU): aggressive splits targeting ~2.5 acts/shard
    //     to fill underutilized CUs.
    //     Verified: FFN Down 7B TP=2→KB=14, TP=4→KB=7.
    //
    // Occupancy floor: ensure at least 8 waves per CU (480 total on 60 CUs)
    // to avoid leaving CUs idle on smaller grid_n shapes.
    constexpr int SMALL_K_THRESHOLD = 16;
    constexpr int MIN_WAVES_PER_CU = 8;
    constexpr int NUM_CUS = 60;

    const int acts_per_wave_single_kb = act_blocks / wk;

    int ideal_kb;
    if (acts_per_wave_single_kb <= SMALL_K_THRESHOLD)
    {
        // Small K: maximize splits to fill CUs
        if (grid_n >= NUM_CUS * 2)
        {
            // Very large grid_n — KB=1 already provides 2+ blocks/CU.
            // Additional splits only increase atomicAdd contention.
            ideal_kb = 1;
        }
        else
        {
            // Target walk ≈ 1 act_block per wave-shard → KB ≈ acts_per_wave.
            // Use ceiling division to avoid pathological KB values where
            // act_blocks isn't evenly divisible by wk. E.g. 14B FFN Down TP=4:
            //   act_blocks=108, wk=8: floor=13 (45.8µs), ceil=14 (35.2µs, -23%).
            //   KB=13 creates uneven shard sizes causing work imbalance.
            // Also fixes 7B Wo TP=4 (28/8: floor=3→20µs, ceil=4→18.7µs).
            ideal_kb = (act_blocks + wk - 1) / wk;
        }
    }
    else
    {
        // Large K: three-branch approach based on CU occupancy at KB=1.
        const float blocks_per_cu_at_kb1 = static_cast<float>(grid_n) / NUM_CUS;

        if (blocks_per_cu_at_kb1 >= 3.0f)
        {
            // High N-parallelism: grid_n fills CUs well but K-walk is long.
            // Moderate splitting reduces the critical path through K-dimension.
            // Target ~5 act_blocks per wave-shard.
            // E.g. 32B FFN Gate TP=1: grid_n=216, apw=20 → KB=4, 12.5% faster.
            ideal_kb = max(1, acts_per_wave_single_kb / 5);
        }
        else
        {
            // Moderate/low N-parallelism: conservative → aggressive two-tier.
            // Tier 1: conservative splits with each wave-shard doing ~16 act_blocks.
            int conservative_kb = max(1, acts_per_wave_single_kb / SMALL_K_THRESHOLD);

            // For very large K (apw > 64), cap conservative KB to limit atomic
            // contention. At these K sizes, each shard is already bandwidth-saturated;
            // more splits just add atomicAdd overhead without improving throughput.
            // Verified: 32B FFN Down TP=1 (apw=108): KB=6→4 saves 10.2%.
            if (acts_per_wave_single_kb > 64)
            {
                conservative_kb = min(conservative_kb, 4);
            }

            int conservative_blocks = grid_n * conservative_kb;

            if (conservative_blocks >= NUM_CUS * 3 / 2)
            {
                // Tier 1 already fills CUs (≥1.5 blocks/CU). Use it.
                // E.g. 14B FFN Gate TP=1: KB=1, blocks=108, 1.8/CU → near-optimal.
                // E.g. FFN Down TP=1: KB=6, blocks=240, 4.0/CU → good.
                ideal_kb = conservative_kb;
            }
            else
            {
                // Tier 2: conservative splits leave CUs underutilized.
                // Target ~2.5 act_blocks per wave-shard for maximum CU parallelism.
                // E.g. FFN Down TP=2: conservative KB=2, blocks=56, only 0.93/CU → bad.
                //       Aggressive KB=14, blocks=392, 6.5/CU → good.
                ideal_kb = (act_blocks * 2) / (wk * 5);
            }
        }
    }

    // Floor: fill CUs with enough waves for memory latency hiding.
    // ONLY for small-K shapes where individual blocks lack sufficient work.
    // For large-K (apw > SMALL_K_THRESHOLD), each block is bandwidth-saturated;
    // the occupancy floor just adds atomic contention without improving utilization.
    // Verified: 14B/32B KV TP=2 (grid_n=4, apw=20): KB=15→8 saves 6-8%.
    if (acts_per_wave_single_kb <= SMALL_K_THRESHOLD)
    {
        const int min_total_waves = MIN_WAVES_PER_CU * NUM_CUS;
        const int min_kb = (min_total_waves + grid_n * wk - 1) / (grid_n * wk);
        ideal_kb = max(ideal_kb, min_kb);
    }

    // Cap: ensure each wave-shard gets ≥1 act_block of work on average.
    // Use ceiling division so fractional remainders (e.g. 28/8=3.5) round up
    // to allow one extra split. Without this, Wo TP=4 (act_blocks=28, wk=8)
    // would cap at KB=3 instead of the sweep-optimal KB=4 (7.3% faster).
    const int kb_max = max(1, (act_blocks + wk - 1) / wk);  // ceil(act_blocks / wk)
    ideal_kb = min(ideal_kb, kb_max);

    return max(1, min(ideal_kb, act_blocks));
}

constexpr int GEMV_INT8_VNNI_ACT_BLOCK_K = 32;
constexpr int GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK = GEMV_INT8_VNNI_ACT_BLOCK_K / 4;

template <int CPT, int KGROUPS_PER_ACT = GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK>
__device__ __forceinline__ void gemv_int8_accumulate_act_block(
    int32_t (&acc)[CPT],
    const int8_t* __restrict__ d_A_int8,
    const int32_t* __restrict__ b_row,
    size_t b_stride,
    int kg_base)
{
    #pragma unroll
    for (int kgi = 0; kgi < KGROUPS_PER_ACT; ++kgi)
    {
        const int kg = kg_base + kgi;
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        if constexpr (CPT == 2)
        {
            const int2 b_pair = *reinterpret_cast<const int2*>(b_row + kg * b_stride);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
            acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int32_t b_packed = *(b_row + kg * b_stride + c);
                acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
            }
        }
    }
}

template <int CPT>
__device__ __forceinline__ void gemv_int8_flush_blockwise_acc(
    int32_t (&block_acc)[CPT],
    float (&fp_acc)[CPT],
    const float* __restrict__ d_scale_A_blockwise,
    int act_block_idx)
{
    const float scale_a = d_scale_A_blockwise[act_block_idx];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        fp_acc[c] += static_cast<float>(block_acc[c]) * scale_a;
        block_acc[c] = 0;
    }
}

__device__ __forceinline__ int gemv_find_batch_for_block(
    const GemvBatchArgs& args,
    int global_block_x)
{
    int batch = 0;
    #pragma unroll
    for (int b = 1; b < MAX_GEMV_BATCH; ++b)
    {
        if (b < args.num_batches && global_block_x >= args.grid_n_offset[b])
        {
            batch = b;
        }
    }
    return batch;
}

template <int TILE_N, int CPT>
__global__ void gemv_init_output_kernel_t(
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias,
    int N,
    float beta)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
        {
            return;
        }

        float result = 0.0f;
        if (beta != 0.0f && d_C_existing)
        {
            result += beta * d_C_existing[n];
        }
        if (d_bias)
        {
            result += d_bias[n];
        }
        d_C_fp32[n] = result;
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_init_output_batched_kernel_t(
    GemvBatchArgs args,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= args.N[batch])
        {
            return;
        }

        float result = 0.0f;
        if (beta != 0.0f)
        {
            result += beta * args.d_C[batch][n];
        }
        if (args.d_bias[batch])
        {
            result += args.d_bias[batch][n];
        }
        args.d_C[batch][n] = result;
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_init_output_pair_kernel_t(
    float* __restrict__ d_C0,
    float* __restrict__ d_C1,
    const float* __restrict__ d_bias0,
    const float* __restrict__ d_bias1,
    int N0,
    int N1,
    int grid_n0,
    float beta)
{
    const bool second = (blockIdx.x >= grid_n0);
    const int local_block_x = second ? (blockIdx.x - grid_n0) : blockIdx.x;
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    float* __restrict__ d_C = second ? d_C1 : d_C0;
    const float* __restrict__ d_bias = second ? d_bias1 : d_bias0;
    const int N = second ? N1 : N0;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N)
        {
            return;
        }

        float result = 0.0f;
        if (beta != 0.0f)
        {
            result += beta * d_C[n];
        }
        if (d_bias)
        {
            result += d_bias[n];
        }
        d_C[n] = result;
    }
}

__global__ void gemv_int8_int8_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
    }

    d_C_int32[n] = acc;
}

/**
 * @brief 2D INT8 VNNI prefill kernel (M>1): computes C[M,N] in one launch.
 *
 * Thread mapping:
 * - threadIdx.y / blockIdx.y choose output row `m`
 * - threadIdx.x / blockIdx.x choose output column `n`
 *
 * For each output element C[m,n], we iterate over K in groups of 4 bytes and
 * use `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) for efficient INT8 MAC.
 *
 * Layout reminder for junior developers:
 * - A is row-major [M x K]
 * - B is VNNI-packed [K/4 x N x 4]
 * - C is row-major [M x N]
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    const int k_groups = K / 4;

    // Compute one output element by reducing across K.
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
        }
    }

    if constexpr (CPT == 4)
    {
        if ((n_base + 3) < N)
        {
            *reinterpret_cast<uint4*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint4(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]), static_cast<uint32_t>(acc[2]), static_cast<uint32_t>(acc[3]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if ((n_base + 1) < N)
        {
            *reinterpret_cast<uint2*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint2(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else
    {
        d_C_int32[static_cast<int64_t>(m) * N + n_base] = acc[0];
    }
}

/**
 * @brief 2D INT8 VNNI prefill kernel with grid-kpar split-K accumulation.
 *
 * grid.z partitions K into slices. Each z-slice computes a partial sum for
 * C[m,n] and atomically accumulates into INT32 output.
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    const int split_idx = blockIdx.z;

    if (m >= M || n_base >= N)
    {
        return;
    }

    const int total_k_groups = K / 4;
    const int k_begin = split_idx * k_groups_per_slice;
    if (k_begin >= total_k_groups)
    {
        return;
    }

    const int k_end = min(total_k_groups, k_begin + k_groups_per_slice);
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int kg = k_begin; kg < k_end; ++kg)
    {
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + kg * 4);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(
                    d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                            d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n) * 4);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }
        }
        else
        {
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<int64_t>(kg) * N + n_base) * 4);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N && acc[c] != 0)
        {
            atomicAdd(&d_C_int32[static_cast<int64_t>(m) * N + n], acc[c]);
        }
    }
}


/**
 * @brief 2D payload-v2 bitwidth-8 prefill kernel (M>1): computes C[M,N].
 *
 * Layout:
 * - A is row-major INT8 [M x K]
 * - Payload is [K/32 x N x 32 bytes] where each record stores 32 signed INT8
 *   values for one (block, output-column)
 * - C is row-major INT32 [M x N]
 */
__global__ void gemm_payload_v2_i8_prefill_kernel(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    const bool valid_n = (n < N);
    const bool valid_m = (m < M);

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;
    extern __shared__ int32_t shared_payload[]; // [blockDim.x * 8]

    for (int b = b0; b < b1; ++b)
    {
        if (threadIdx.y == 0)
        {
            int32_t* payload_dst = shared_payload + static_cast<int>(threadIdx.x) * 8;
            if (valid_n)
            {
                const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
                const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);
                payload_dst[0] = static_cast<int32_t>(packed0.x);
                payload_dst[1] = static_cast<int32_t>(packed0.y);
                payload_dst[2] = static_cast<int32_t>(packed0.z);
                payload_dst[3] = static_cast<int32_t>(packed0.w);
                payload_dst[4] = static_cast<int32_t>(packed1.x);
                payload_dst[5] = static_cast<int32_t>(packed1.y);
                payload_dst[6] = static_cast<int32_t>(packed1.z);
                payload_dst[7] = static_cast<int32_t>(packed1.w);
            }
            else
            {
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    payload_dst[g] = 0;
                }
            }
        }

        __syncthreads();

        if (valid_m && valid_n)
        {
            const int32_t* a4 = reinterpret_cast<const int32_t*>(
                d_A_int8 + static_cast<int64_t>(m) * K + b * 32);
            const int32_t* payload_i32 = shared_payload + static_cast<int>(threadIdx.x) * 8;

            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc, false);
            }
        }

        __syncthreads();
    }

    if (valid_m && valid_n)
    {
        const int64_t out_idx = static_cast<int64_t>(m) * N + n;
        if (kb == 1)
        {
            d_C_int32[out_idx] = acc;
        }
        else
        {
            atomicAdd(&d_C_int32[out_idx], acc);
        }
    }
}

__global__ void gemm_payload_v2_i8_prefill_kernel_direct(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc = 0;

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
        const uint4 packed0 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u);
        const uint4 packed1 = *reinterpret_cast<const uint4*>(d_payload + linear * 32u + 16u);

        acc = __builtin_amdgcn_sdot4(a4[0], static_cast<int32_t>(packed0.x), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[1], static_cast<int32_t>(packed0.y), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[2], static_cast<int32_t>(packed0.z), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[3], static_cast<int32_t>(packed0.w), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[4], static_cast<int32_t>(packed1.x), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[5], static_cast<int32_t>(packed1.y), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[6], static_cast<int32_t>(packed1.z), acc, false);
        acc = __builtin_amdgcn_sdot4(a4[7], static_cast<int32_t>(packed1.w), acc, false);
    }

    const int64_t out_idx = static_cast<int64_t>(m) * N + n;
    if (kb == 1)
    {
        d_C_int32[out_idx] = acc;
    }
    else
    {
        atomicAdd(&d_C_int32[out_idx], acc);
    }
}

template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void gemm_payload_v2_i8_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const uint8_t* __restrict__ d_payload,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int kblocks)
{
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (m >= M || n_base >= N)
    {
        return;
    }

    const int blocks_per_row = K / 32;
    const int kb = max(1, kblocks);
    const int b_per = (blocks_per_row + kb - 1) / kb;
    const int b0 = min(blocks_per_row, static_cast<int>(blockIdx.z) * b_per);
    const int b1 = min(blocks_per_row, b0 + b_per);

    if (b0 >= b1)
    {
        return;
    }

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    for (int b = b0; b < b1; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K + b * 32);

        const size_t linear0 = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n_base);
        const int32_t* payload_i32_0 = reinterpret_cast<const int32_t*>(d_payload + linear0 * 32u);

        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                const int32_t* payload_i32_2 = payload_i32_1 + 8;
                const int32_t* payload_i32_3 = payload_i32_2 + 8;

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = __builtin_amdgcn_sdot4(a, payload_i32_0[g], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a, payload_i32_1[g], acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a, payload_i32_2[g], acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a, payload_i32_3[g], acc[3], false);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc[c], false);
                        }
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                const int32_t* payload_i32_1 = payload_i32_0 + 8;
                #pragma unroll
                for (int g = 0; g < 8; ++g)
                {
                    const int32_t a = a4[g];
                    acc[0] = __builtin_amdgcn_sdot4(a, payload_i32_0[g], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a, payload_i32_1[g], acc[1], false);
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                        const int32_t* payload_i32 = reinterpret_cast<const int32_t*>(d_payload + linear * 32u);
                        #pragma unroll
                        for (int g = 0; g < 8; ++g)
                        {
                            acc[c] = __builtin_amdgcn_sdot4(a4[g], payload_i32[g], acc[c], false);
                        }
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                acc[0] = __builtin_amdgcn_sdot4(a4[g], payload_i32_0[g], acc[0], false);
            }
        }
    }

    const int64_t out_base = static_cast<int64_t>(m) * N + n_base;
    if (kb == 1)
    {
        if constexpr (CPT == 4)
        {
            if ((n_base + 3) < N)
            {
                *reinterpret_cast<uint4*>(d_C_int32 + out_base) = make_uint4(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]),
                    static_cast<uint32_t>(acc[2]),
                    static_cast<uint32_t>(acc[3]));
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int n = n_base + c;
                    if (n < N)
                    {
                        d_C_int32[out_base + c] = acc[c];
                    }
                }
            }
        }
        else if constexpr (CPT == 2)
        {
            if ((n_base + 1) < N)
            {
                *reinterpret_cast<uint2*>(d_C_int32 + out_base) = make_uint2(
                    static_cast<uint32_t>(acc[0]),
                    static_cast<uint32_t>(acc[1]));
            }
            else
            {
                d_C_int32[out_base] = acc[0];
                if ((n_base + 1) < N)
                {
                    d_C_int32[out_base + 1] = acc[1];
                }
            }
        }
        else
        {
            d_C_int32[out_base] = acc[0];
        }
    }
    else
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                atomicAdd(&d_C_int32[out_base + c], acc[c]);
            }
        }
    }
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base) * 4);

        if constexpr (VEC_PACKED_LOAD && CPT == 4)
        {
            if (full_cols)
            {
                const int4 b = *reinterpret_cast<const int4*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, acc[3], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else if constexpr (VEC_PACKED_LOAD && CPT == 2)
        {
            if (full_cols)
            {
                const int2 b = *reinterpret_cast<const int2*>(b_row);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                if (n_base + c < N)
                {
                    const int32_t b = *(b_row + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                }
            }
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        if (n_base + c < N)
        {
            d_C_int32[n_base + c] = acc[c];
        }
    }
}

// Wide-N kernel: 2 columns per thread
__global__ void gemv_int8_int8_wide2_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n2 = blockIdx.x * GEMV_INT8_VNNI_WIDE2_TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
        const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

        // Load 4 INT8 weights for each of 2 columns (VNNI = already packed)
        const int32_t p0 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 0)) * 4);
        const int32_t p1 = *reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 1)) * 4);

        // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
        acc0 = __builtin_amdgcn_sdot4(a_packed, p0, acc0, false);
        acc1 = __builtin_amdgcn_sdot4(a_packed, p1, acc1, false);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// K-parallel kernel: scalar threads, shared A tiles
__global__ void gemv_int8_int8_square_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int tile_groups = tile_k / 4;
        const int k_group_base = k_base / 4;
        for (int kg = 0; kg < tile_groups; ++kg)
        {
            // Load 4 INT8 activations from shared memory as packed int32
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + kg * 4);

            const int global_kg = k_group_base + kg;
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);

            // Hardware INT8 dot product: 4 multiply-adds in 1 cycle (v_dot4_i32_i8)
            acc = __builtin_amdgcn_sdot4(a_packed, b_packed, acc, false);
        }
        __syncthreads();
    }

    d_C_int32[n] = acc;
}

__global__ void gemv_int8_int8_vnni_blockwise_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    float fp_acc = 0.0f;
    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t block_acc = 0;
        const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
        #pragma unroll
        for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
        {
            const int kg = kg_base + kgi;
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
            const int32_t b_packed = *(b_row + kg * b_stride);
            block_acc = __builtin_amdgcn_sdot4(a_packed, b_packed, block_acc, false);
        }
        fp_acc += static_cast<float>(block_acc) * d_scales_A_blockwise[act_block];
    }

    float result = fp_acc * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    const bool full_cols = (n_base + (CPT - 1) < N);
    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n_base) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0;

        const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
        #pragma unroll
        for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
        {
            const int kg = kg_base + kgi;
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
            const int32_t* __restrict__ b_pack_row = b_row + kg * b_stride;

            if constexpr (VEC_PACKED_LOAD && CPT == 4)
            {
                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_pack_row);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, acc[3], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                        }
                    }
                }
            }
            else if constexpr (VEC_PACKED_LOAD && CPT == 2)
            {
                if (full_cols)
                {
                    const int2 b = *reinterpret_cast<const int2*>(b_pack_row);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                        }
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_pack_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }

        const float scale_a = d_scales_A_blockwise[act_block];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            fp_acc[c] += static_cast<float>(acc[c]) * scale_a;
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float out = fp_acc[c] * d_scale_B[n];
            out *= alpha;
            if (beta != 0.0f)
            {
                out += beta * d_C_existing[n];
            }
            if (d_bias)
            {
                out += d_bias[n];
            }
            d_C_fp32[n] = out;
        }
    }
}

__global__ void gemv_int8_int8_square_vnni_blockwise_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    float alpha,
    float beta,
    const float* __restrict__ d_C_existing,
    const float* __restrict__ d_bias)
{
    const int n = blockIdx.x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= N) return;

    float fp_acc = 0.0f;
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_tile_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_tile_base + i];
        }
        __syncthreads();

        const int tile_act_blocks = tile_k / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int act_block_base = k_tile_base / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int tile_k_group_base = k_tile_base / 4;

        #pragma unroll
        for (int tile_block = 0; tile_block < GEMV_INT8_VNNI_KPAR_TILE_K / GEMV_INT8_VNNI_ACT_BLOCK_K; ++tile_block)
        {
            if (tile_block >= tile_act_blocks)
            {
                break;
            }

            int32_t block_acc = 0;
            const int local_k_byte = tile_block * GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int k_group_base = tile_k_group_base + tile_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + local_k_byte + kgi * 4);
                const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(k_group_base + kgi) * N + n) * 4);
                block_acc = __builtin_amdgcn_sdot4(a_packed, b_packed, block_acc, false);
            }

            fp_acc += static_cast<float>(block_acc) * d_scales_A_blockwise[act_block_base + tile_block];
        }
        __syncthreads();
    }

    float result = fp_acc * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_existing[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_batched_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    float alpha,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    if (n_base >= args.N[batch]) return;

    const int N = args.N[batch];
    const int8_t* __restrict__ d_B_int8_vnni = args.d_B[batch];
    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const float* __restrict__ d_bias = args.d_bias[batch];
    const bool full_cols = (n_base + (CPT - 1) < N);
    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n_base) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0;

        gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                       act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);

        const float scale_a = d_scales_A_blockwise[act_block];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            fp_acc[c] += static_cast<float>(acc[c]) * scale_a;
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float result = fp_acc[c] * d_scale_B[n];
            result *= alpha;
            if (beta != 0.0f)
            {
                result += beta * d_C_fp32[n];
            }
            if (d_bias)
            {
                result += d_bias[n];
            }
            d_C_fp32[n] = result;
        }
    }
}

template <int TILE_N, int CPT, bool VEC_PACKED_LOAD>
__global__ void gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    float alpha,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    if (n_base >= args.N[batch]) return;

    const int N = args.N[batch];
    const int8_t* __restrict__ d_B_int8_vnni = args.d_B[batch];
    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const float* __restrict__ d_bias = args.d_bias[batch];
    const bool full_cols = (n_base + (CPT - 1) < N);

    const int32_t* __restrict__ b_row = reinterpret_cast<const int32_t*>(
        d_B_int8_vnni + static_cast<size_t>(n_base) * 4);
    const size_t b_stride = static_cast<size_t>(N);
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    for (int act_block = 0; act_block < act_blocks; ++act_block)
    {
        int32_t acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) acc[c] = 0;

        const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
        #pragma unroll
        for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
        {
            const int kg = kg_base + kgi;
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);
            const int32_t* __restrict__ b_pack_row = b_row + kg * b_stride;

            if constexpr (VEC_PACKED_LOAD && CPT == 4)
            {
                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_pack_row);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, acc[3], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                        }
                    }
                }
            }
            else if constexpr (VEC_PACKED_LOAD && CPT == 2)
            {
                if (full_cols)
                {
                    const int2 b = *reinterpret_cast<const int2*>(b_pack_row);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                        }
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b = *(b_pack_row + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b, acc[c], false);
                    }
                }
            }
        }

        const float scale_a = d_scales_A_blockwise[act_block];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            fp_acc[c] += static_cast<float>(acc[c]) * scale_a;
        }
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float result = fp_acc[c] * d_scale_B[n];
            result *= alpha;
            if (beta != 0.0f)
            {
                result += beta * d_C_fp32[n];
            }
            if (d_bias)
            {
                result += d_bias[n];
            }
            d_C_fp32[n] = result;
        }
    }
}

__global__ void gemv_int8_int8_square_batched_vnni_blockwise_scaled_kernel(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    float alpha,
    float beta)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n = local_block_x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= args.N[batch]) return;

    const int8_t* __restrict__ d_B_int8_vnni = args.d_B[batch];
    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const float* __restrict__ d_bias = args.d_bias[batch];
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    float fp_acc = 0.0f;
    for (int k_tile_base = 0; k_tile_base < K; k_tile_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_tile_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_tile_base + i];
        }
        __syncthreads();

        const int tile_act_blocks = tile_k / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int act_block_base = k_tile_base / GEMV_INT8_VNNI_ACT_BLOCK_K;
        const int tile_k_group_base = k_tile_base / 4;

        #pragma unroll
        for (int tile_block = 0; tile_block < GEMV_INT8_VNNI_KPAR_TILE_K / GEMV_INT8_VNNI_ACT_BLOCK_K; ++tile_block)
        {
            if (tile_block >= tile_act_blocks)
            {
                break;
            }

            int32_t block_acc = 0;
            const int local_k_byte = tile_block * GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int k_group_base = tile_k_group_base + tile_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + local_k_byte + kgi * 4);
                const int32_t b_packed = *reinterpret_cast<const int32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(k_group_base + kgi) * args.N[batch] + n) * 4);
                block_acc = __builtin_amdgcn_sdot4(a_packed, b_packed, block_acc, false);
            }

            fp_acc += static_cast<float>(block_acc) * d_scales_A_blockwise[act_block_base + tile_block];
        }
        __syncthreads();
    }

    float result = fp_acc * d_scale_B[n];
    result *= alpha;
    if (beta != 0.0f)
    {
        result += beta * d_C_fp32[n];
    }
    if (d_bias)
    {
        result += d_bias[n];
    }
    d_C_fp32[n] = result;
}

// ============================================================================
// Grid-level K-parallel VNNI kernel with atomicAdd reduction
//
// Each block processes a TILE_N/CPT-thread tile × one K-segment.
// Grid: (ceil(N/TILE_N), KB) where KB splits K across blockIdx.y.
// Output MUST be zeroed before launch (atomicAdd accumulation).
//
// Key insight: Grid-level K-splitting creates 28×48=1344 independent blocks
// (vs 56 with square_vnni), providing enough wavefronts to fully hide
// HBM2 latency on MI50 (60 CUs × 40 max concurrent waves = 2400).
//
// Benchmark results (MI50, gfx906):
//   Q/Wo (3584×3584):    22.0μs / 58.5% peak BW (9.6× vs square_vnni)
//   FFN Down (3584×18944): 95.7μs / 70.9% peak BW (2.0× vs smem kpar)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    // Each block processes its K-segment: [k_start, k_end)
    // CRITICAL: k_per must be a multiple of 4 to ensure k_start aligns with
    // VNNI 4-byte groups. If k_start is not a multiple of 4, kg_offset would
    // misalign with the weight K-groups, producing garbage output.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;  // Round up to multiple of 4
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // Accumulate partial dot product for CPT output columns
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        // Process VNNI k-groups within this block's K-segment
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;

        // Precompute B pointer: b_row[kg] points to the start of this thread's
        // columns for k-group kg. b_row = B_int32_ptr + n_base, stride = N (int32 units).
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Partial unroll: expose 4 k-groups per iteration for better
        // memory-level parallelism and instruction scheduling
        #pragma unroll 4
        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            // Load 4 INT8 activations as packed int32 for v_dot4_i32_i8
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                // Vectorized 8-byte load: two adjacent VNNI columns in one
                // memory transaction. n_base is always even (threadIdx.x * 2),
                // so int2 alignment is guaranteed for N divisible by 2.
                const int2 b_pair = *reinterpret_cast<const int2*>(
                    b_row + kg * b_stride);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
            }
            else
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c)
                {
                    const int32_t b_packed = *(b_row + kg * b_stride + c);
                    acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                }
            }
        }

        // Accumulate partials to output via atomicAdd
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int act_block_splits,
    float alpha)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int split_idx = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int act_blocks_per_split = (act_blocks + act_block_splits - 1) / act_block_splits;
    const int act_block_start = split_idx * act_blocks_per_split;
    const int act_block_end = min(act_block_start + act_blocks_per_split, act_blocks);

    if (!active || act_block_start >= act_blocks)
    {
        return;
    }

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    const int32_t* __restrict__ b_row =
        reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
    const size_t b_stride = static_cast<size_t>(N);
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    constexpr int ACT_BLOCKS_PER_TILE = GEMV_INT8_VNNI_KPAR_TILE_K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    for (int act_block_tile = act_block_start; act_block_tile < act_block_end; act_block_tile += ACT_BLOCKS_PER_TILE)
    {
        const int tile_act_blocks = min(ACT_BLOCKS_PER_TILE, act_block_end - act_block_tile);
        const int tile_bytes = tile_act_blocks * GEMV_INT8_VNNI_ACT_BLOCK_K;
        for (int i = threadIdx.x; i < tile_bytes; i += blockDim.x)
        {
            sA[i] = d_A_int8[act_block_tile * GEMV_INT8_VNNI_ACT_BLOCK_K + i];
        }
        __syncthreads();

        #pragma unroll
        for (int tile_block = 0; tile_block < ACT_BLOCKS_PER_TILE; ++tile_block)
        {
            if (tile_block >= tile_act_blocks)
            {
                break;
            }

            int32_t block_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                block_acc[c] = 0;
            }

            const int local_k_byte = tile_block * GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int kg_base = (act_block_tile + tile_block) * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(sA + local_k_byte + kgi * 4);
                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(b_row + (kg_base + kgi) * b_stride);
                    block_acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, block_acc[0], false);
                    block_acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, block_acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_base + kgi) * b_stride + c);
                        block_acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, block_acc[c], false);
                    }
                }
            }

            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block_tile + tile_block);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            const float val = fp_acc[c] * d_scale_B[n] * alpha;
            if (act_block_splits == 1)
            {
                d_C_fp32[n] = val;
            }
            else
            {
                atomicAdd(&d_C_fp32[n], val);
            }
        }
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_batched_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    GemvBatchArgs args,
    int K,
    int act_block_splits,
    float alpha)
{
    const int batch = gemv_find_batch_for_block(args, blockIdx.x);
    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    const int N = args.N[batch];
    const bool active = (n_base < N);
    const int split_idx = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int act_blocks_per_split = (act_blocks + act_block_splits - 1) / act_block_splits;
    const int act_block_start = split_idx * act_blocks_per_split;
    const int act_block_end = min(act_block_start + act_blocks_per_split, act_blocks);

    if (!active || act_block_start >= act_blocks)
    {
        return;
    }

    float* __restrict__ d_C_fp32 = args.d_C[batch];
    const float* __restrict__ d_scale_B = args.d_scales_B[batch];
    const int32_t* __restrict__ b_row =
        reinterpret_cast<const int32_t*>(args.d_B[batch]) + n_base;
    const size_t b_stride = static_cast<size_t>(N);

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
    {
        int32_t block_acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            block_acc[c] = 0;
        }

        gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                       act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
        gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            atomicAdd(&d_C_fp32[n], fp_acc[c] * d_scale_B[n] * alpha);
        }
    }
}

template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_pair_vnni_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const float* __restrict__ d_scales_A_blockwise,
    const int8_t* __restrict__ d_B0,
    const int8_t* __restrict__ d_B1,
    float* __restrict__ d_C0,
    float* __restrict__ d_C1,
    const float* __restrict__ d_scale_B0,
    const float* __restrict__ d_scale_B1,
    int N0,
    int N1,
    int grid_n0,
    int K,
    int act_block_splits,
    float alpha)
{
    const bool second = (blockIdx.x >= grid_n0);
    const int local_block_x = second ? (blockIdx.x - grid_n0) : blockIdx.x;
    const int n_base = local_block_x * TILE_N + threadIdx.x * CPT;
    const int N = second ? N1 : N0;
    const bool active = (n_base < N);
    const int split_idx = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int act_blocks_per_split = (act_blocks + act_block_splits - 1) / act_block_splits;
    const int act_block_start = split_idx * act_blocks_per_split;
    const int act_block_end = min(act_block_start + act_blocks_per_split, act_blocks);

    if (!active || act_block_start >= act_blocks)
    {
        return;
    }

    float* __restrict__ d_C_fp32 = second ? d_C1 : d_C0;
    const float* __restrict__ d_scale_B = second ? d_scale_B1 : d_scale_B0;
    const int8_t* __restrict__ d_B_int8_vnni = second ? d_B1 : d_B0;
    const int32_t* __restrict__ b_row =
        reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
    const size_t b_stride = static_cast<size_t>(N);

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
    {
        int32_t block_acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            block_acc[c] = 0;
        }

        gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                       act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
        gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
    }

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            atomicAdd(&d_C_fp32[n], fp_acc[c] * d_scale_B[n] * alpha);
        }
    }
}


// ============================================================================
// LDS K-Reduction GEMV kernel: eliminates atomicAdd via intra-block reduction
//
// KEY INSIGHT: Instead of using 1-wavefront blocks with grid-level atomicAdd,
// use WK-wavefront blocks where each wavefront computes a K-slice and the
// partial sums are reduced via LDS within the block.
//
// Advantages over grid_kpar:
//   1. Eliminates atomicAdd when kb_outer==1 (large-N shapes like FFN)
//   2. Eliminates hipMemsetAsync when kb_outer==1
//   3. Reduces atomicAdd count by WK× when kb_outer>1 (small-N shapes)
//   4. Same total wavefront count → same occupancy and memory bandwidth
//
// Thread layout:
//   Block: WK × (TILE_N / CPT) threads = 4 × 64 = 256
//   wave_id = threadIdx.x / (TILE_N/CPT) → K-slice index (0..WK-1)
//   lane_id = threadIdx.x % (TILE_N/CPT) → column index within tile
//   Each wave handles TILE_N output columns across K/(WK*kb_outer) k-groups
//
// LDS usage: (WK-1) × TILE_N × sizeof(int32_t)
//   WK=4, TILE_N=128: 3 × 128 × 4 = 1536 bytes (trivial)
//
// Grid: (ceil(N/TILE_N), kb_outer)
//   When kb_outer==1: plain store (no atomicAdd, no memset)
//   When kb_outer>1:  atomicAdd (but WK× fewer than grid_kpar)
//
// Benchmark motivation (MI50, gfx906):
//   FFN Gate/Up (18944×3584): grid_kpar KB=4 → 756 GB/s (76% peak)
//     - 148 N-blocks × 4 K-blocks = 592 wavefronts = 9.9 waves/CU
//     - 18944 × 4 = 75776 atomicAdds per launch
//   NEW lds_kreduce kb_outer=1: same 592 wavefronts, zero atomicAdds
//   Q/Wo (3584×3584): grid_kpar KB=16 → 456 GB/s (46% peak)
//     - 28 × 16 = 448 wavefronts, 3584 × 16 = 57344 atomicAdds
//   NEW lds_kreduce kb_outer=4: 28×4×4 = 448 wavefronts, 3584 × 4 atomicAdds
// ============================================================================
template <int TILE_N, int CPT, int WK>
__global__ __launch_bounds__(TILE_N / CPT * WK)
void gemv_int8_int8_lds_kreduce_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kb_outer)
{
    static_assert(CPT == 2 || CPT == 4, "CPT must be 2 or 4");
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const int outer_k_block = blockIdx.y;

    // --- Compute K-range for this block's outer K-slice ---
    const int total_k_groups = K / 4;

    // Evenly distribute k-groups across kb_outer grid blocks
    const int kgroups_per_outer = (total_k_groups + kb_outer - 1) / kb_outer;
    const int outer_kg_start = outer_k_block * kgroups_per_outer;
    const int outer_kg_end = min(outer_kg_start + kgroups_per_outer, total_k_groups);
    if (outer_kg_start >= total_k_groups) return;

    const int outer_kgroup_count = outer_kg_end - outer_kg_start;

    // --- Inner K-split: distribute outer range across WK wavefronts ---
    // Balanced distribution: waves 0..remainder-1 get one extra k-group
    const int kg_per_wave_base = outer_kgroup_count / WK;
    const int kg_remainder = outer_kgroup_count % WK;
    const int my_kg_start = outer_kg_start
        + wave_id * kg_per_wave_base
        + min(wave_id, kg_remainder);
    const int my_kg_count = kg_per_wave_base + (wave_id < kg_remainder ? 1 : 0);
    const int my_kg_end = my_kg_start + my_kg_count;

    // Accumulators — each thread accumulates CPT output column values
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    // Main K-reduction loop (innermost hot loop)
    if (n_base < N && my_kg_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // #pragma unroll 4: expose 4 k-groups for memory-level parallelism.
        // Compiler generates 4× global_load + 4× s_load + 8× v_dot4 interleaved.
        #pragma unroll 4
        for (int kg = my_kg_start; kg < my_kg_end; ++kg)
        {
            // A is uniform across all threads in this wave (M=1)
            // → compiler uses scalar load (s_load_dword), broadcast to all lanes
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

            if constexpr (CPT == 2)
            {
                // 8-byte vectorized load: two adjacent VNNI columns
                const int2 b_pair = *reinterpret_cast<const int2*>(
                    b_row + kg * b_stride);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
            }
            else // CPT == 4
            {
                // 16-byte vectorized load: four adjacent VNNI columns
                const int4 b_pack = *reinterpret_cast<const int4*>(
                    b_row + kg * b_stride);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pack.x, acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pack.y, acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b_pack.z, acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b_pack.w, acc[3], false);
            }
        }
    }

    // =========================================================================
    // LDS K-reduction: sum partial results across WK wavefronts
    //
    // Waves 1..WK-1 deposit their partial sums into LDS.
    // After barrier, wave 0 reads and sums all partials + its own accumulator.
    // This replaces global atomicAdd with fast LDS reads (5-10 cycles each).
    // =========================================================================
    __shared__ int32_t s_partial[WK - 1][TILE_N];

    // Waves 1..WK-1 write partials to LDS
    if (wave_id > 0 && n_base < N)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            s_partial[wave_id - 1][lane_id * CPT + c] = acc[c];
        }
    }

    __syncthreads();

    // Wave 0 sums all partials and writes to global output
    if (wave_id == 0 && n_base < N)
    {
        // Sum own partial + (WK-1) LDS contributions
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                acc[c] += s_partial[w][lane_id * CPT + c];
            }
        }

        // Store output
        if (kb_outer == 1)
        {
            // Plain store — no atomicAdd needed, no pre-zeroing required
            if constexpr (CPT == 2)
            {
                *reinterpret_cast<int2*>(&d_C_int32[n_base]) =
                    make_int2(acc[0], acc[1]);
            }
            else // CPT == 4
            {
                *reinterpret_cast<int4*>(&d_C_int32[n_base]) =
                    make_int4(acc[0], acc[1], acc[2], acc[3]);
            }
        }
        else
        {
            // Multi-block outer K-split: atomicAdd (but WK× fewer than grid_kpar)
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                atomicAdd(&d_C_int32[n_base + c], acc[c]);
            }
        }
    }
}

template <int TILE_N, int CPT, int WK>
__global__ __launch_bounds__(TILE_N / CPT * WK)
void gemv_int8_int8_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(CPT == 2 || CPT == 4, "CPT must be 2 or 4");
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 1
        for (int act_block = my_act_start; act_block < my_act_end; ++act_block)
        {
            int32_t block_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) block_acc[c] = 0;

            gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                           act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    __shared__ float s_partial[WK - 1][TILE_N];

    if (wave_id > 0 && n_base < N)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            s_partial[wave_id - 1][lane_id * CPT + c] = fp_acc[c];
        }
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                fp_acc[c] += s_partial[w][lane_id * CPT + c];
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            const float partial = fp_acc[c] * d_scale_B[n] * alpha;
            if (kb_outer == 1)
            {
                d_C_fp32[n] = partial;
            }
            else
            {
                atomicAdd(&d_C_fp32[n], partial);
            }
        }
    }
}

// Pair variant of the tiny K/V LDS K-reduce kernel.
// Dispatches two projections (e.g. K+V) in a single kernel launch
// with combined grid_x = ceil(N0/TN) + ceil(N1/TN).
// Doubles occupancy and saves one launch overhead vs 2x single.
template <int TILE_N, int CPT, int WK>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_lds_kreduce_pair_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B0,
    const int8_t* __restrict__ d_B1,
    float* __restrict__ d_C0,
    float* __restrict__ d_C1,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scales_B0,
    const float* __restrict__ d_scales_B1,
    int N0, int N1,
    int grid_n_offset,
    int K, int kb_outer,
    float alpha)
{
    static_assert(CPT == 2 || CPT == 4, "CPT must be 2 or 4");
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    // Determine which projection this block serves
    const bool is_proj1 = (static_cast<int>(blockIdx.x) >= grid_n_offset);
    const int local_block_x = is_proj1 ? (static_cast<int>(blockIdx.x) - grid_n_offset) : static_cast<int>(blockIdx.x);
    const int8_t* __restrict__ d_B = is_proj1 ? d_B1 : d_B0;
    float* __restrict__ d_C = is_proj1 ? d_C1 : d_C0;
    const float* __restrict__ d_scale_B = is_proj1 ? d_scales_B1 : d_scales_B0;
    const int N = is_proj1 ? N1 : N0;

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = local_block_x * TILE_N + lane_id * CPT;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 1
        for (int act_block = my_act_start; act_block < my_act_end; ++act_block)
        {
            int32_t block_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) block_acc[c] = 0;

            gemv_int8_accumulate_act_block(block_acc, d_A_int8, b_row, b_stride,
                                           act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    __shared__ float s_partial[WK - 1][TILE_N];

    if (wave_id > 0 && n_base < N)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            s_partial[wave_id - 1][lane_id * CPT + c] = fp_acc[c];
        }
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                fp_acc[c] += s_partial[w][lane_id * CPT + c];
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            const float partial = fp_acc[c] * d_scale_B[n] * alpha;
            if (kb_outer == 1)
            {
                d_C[n] = partial;
            }
            else
            {
                atomicAdd(&d_C[n], partial);
            }
        }
    }
}

template <int WK>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");

    constexpr int TILE_N = 256;
    constexpr int CPT = 4;
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const bool full_cols = (n_base + 3) < N;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        #pragma unroll 1
        for (int act_block = my_act_start; act_block < my_act_end; ++act_block)
        {
            int32_t block_acc[4] = {0, 0, 0, 0};
            const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;

            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base + kgi) * 4);
                const int32_t* __restrict__ b_pack_row = b_row + (kg_base + kgi) * b_stride;

                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_pack_row);
                    block_acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, block_acc[0], false);
                    block_acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, block_acc[1], false);
                    block_acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, block_acc[2], false);
                    block_acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, block_acc[3], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t b = *(b_pack_row + c);
                            block_acc[c] = __builtin_amdgcn_sdot4(a_packed, b, block_acc[c], false);
                        }
                    }
                }
            }

            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    __shared__ float4 s_partial[WK - 1][64];

    if (wave_id > 0 && n_base < N)
    {
        s_partial[wave_id - 1][lane_id] = make_float4(fp_acc[0], fp_acc[1], fp_acc[2], fp_acc[3]);
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            const float4 partial = s_partial[w][lane_id];
            fp_acc[0] += partial.x;
            fp_acc[1] += partial.y;
            fp_acc[2] += partial.z;
            fp_acc[3] += partial.w;
        }

        if (full_cols)
        {
            const float4 scale = *reinterpret_cast<const float4*>(d_scale_B + n_base);
            if (kb_outer == 1)
            {
                d_C_fp32[n_base + 0] = fp_acc[0] * scale.x * alpha;
                d_C_fp32[n_base + 1] = fp_acc[1] * scale.y * alpha;
                d_C_fp32[n_base + 2] = fp_acc[2] * scale.z * alpha;
                d_C_fp32[n_base + 3] = fp_acc[3] * scale.w * alpha;
            }
            else
            {
                atomicAdd(&d_C_fp32[n_base + 0], fp_acc[0] * scale.x * alpha);
                atomicAdd(&d_C_fp32[n_base + 1], fp_acc[1] * scale.y * alpha);
                atomicAdd(&d_C_fp32[n_base + 2], fp_acc[2] * scale.z * alpha);
                atomicAdd(&d_C_fp32[n_base + 3], fp_acc[3] * scale.w * alpha);
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const float partial = fp_acc[c] * d_scale_B[n] * alpha;
                    if (kb_outer == 1)
                    {
                        d_C_fp32[n] = partial;
                    }
                    else
                    {
                        atomicAdd(&d_C_fp32[n], partial);
                    }
                }
            }
        }
    }
}


// ============================================================================
// Q/Wo 2x-unrolled LDS K-reduction blockwise kernel
//
// Processes 2 activation blocks simultaneously with separate INT32
// accumulators, interleaving dot products for better ILP. The compiler
// can overlap B loads for block1 with sdot4 computation for block0,
// and pipeline flushes from both blocks.
//
// Register pressure: 2*4 = 8 INT32 acc + 4 FP32 acc = 12 VGPRs for accums.
// ============================================================================
template <int WK>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");

    constexpr int TILE_N = 256;
    constexpr int CPT = 4;
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const bool full_cols = (n_base + 3) < N;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[4];
    #pragma unroll
    for (int c = 0; c < 4; ++c)
    {
        fp_acc[c] = 0.0f;
    }

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Process pairs of activation blocks for ILP
        int act_block = my_act_start;
        const int pair_end = my_act_start + (my_act_count & ~1);  // round down to even

        #pragma unroll 1
        for (; act_block < pair_end; act_block += 2)
        {
            int32_t acc0[4] = {0, 0, 0, 0};
            int32_t acc1[4] = {0, 0, 0, 0};
            const int kg_base0 = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;
            const int kg_base1 = (act_block + 1) * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;

            // Interleave k-groups from both blocks for ILP
            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a0 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base0 + kgi) * 4);
                const int32_t a1 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base1 + kgi) * 4);

                if (full_cols)
                {
                    const int4 b0 = *reinterpret_cast<const int4*>(b_row + (kg_base0 + kgi) * b_stride);
                    acc0[0] = __builtin_amdgcn_sdot4(a0, b0.x, acc0[0], false);
                    acc0[1] = __builtin_amdgcn_sdot4(a0, b0.y, acc0[1], false);
                    acc0[2] = __builtin_amdgcn_sdot4(a0, b0.z, acc0[2], false);
                    acc0[3] = __builtin_amdgcn_sdot4(a0, b0.w, acc0[3], false);

                    const int4 b1 = *reinterpret_cast<const int4*>(b_row + (kg_base1 + kgi) * b_stride);
                    acc1[0] = __builtin_amdgcn_sdot4(a1, b1.x, acc1[0], false);
                    acc1[1] = __builtin_amdgcn_sdot4(a1, b1.y, acc1[1], false);
                    acc1[2] = __builtin_amdgcn_sdot4(a1, b1.z, acc1[2], false);
                    acc1[3] = __builtin_amdgcn_sdot4(a1, b1.w, acc1[3], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t bv0 = *(b_row + (kg_base0 + kgi) * b_stride + c);
                            acc0[c] = __builtin_amdgcn_sdot4(a0, bv0, acc0[c], false);
                            const int32_t bv1 = *(b_row + (kg_base1 + kgi) * b_stride + c);
                            acc1[c] = __builtin_amdgcn_sdot4(a1, bv1, acc1[c], false);
                        }
                    }
                }
            }

            // Flush both blocks
            const float scale0 = d_scales_A_blockwise[act_block];
            const float scale1 = d_scales_A_blockwise[act_block + 1];
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                fp_acc[c] += static_cast<float>(acc0[c]) * scale0;
                fp_acc[c] += static_cast<float>(acc1[c]) * scale1;
            }
        }

        // Handle odd tail (at most 1 remaining block)
        if (act_block < my_act_end)
        {
            int32_t block_acc[4] = {0, 0, 0, 0};
            const int kg_base = act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK;

            #pragma unroll
            for (int kgi = 0; kgi < GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK; ++kgi)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base + kgi) * 4);

                if (full_cols)
                {
                    const int4 b = *reinterpret_cast<const int4*>(b_row + (kg_base + kgi) * b_stride);
                    block_acc[0] = __builtin_amdgcn_sdot4(a_packed, b.x, block_acc[0], false);
                    block_acc[1] = __builtin_amdgcn_sdot4(a_packed, b.y, block_acc[1], false);
                    block_acc[2] = __builtin_amdgcn_sdot4(a_packed, b.z, block_acc[2], false);
                    block_acc[3] = __builtin_amdgcn_sdot4(a_packed, b.w, block_acc[3], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (n_base + c < N)
                        {
                            const int32_t bv = *(b_row + (kg_base + kgi) * b_stride + c);
                            block_acc[c] = __builtin_amdgcn_sdot4(a_packed, bv, block_acc[c], false);
                        }
                    }
                }
            }

            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    // LDS K-reduction: same as standard Q/Wo kernel
    __shared__ float4 s_partial[WK - 1][64];

    if (wave_id > 0 && n_base < N)
    {
        s_partial[wave_id - 1][lane_id] = make_float4(fp_acc[0], fp_acc[1], fp_acc[2], fp_acc[3]);
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            const float4 partial = s_partial[w][lane_id];
            fp_acc[0] += partial.x;
            fp_acc[1] += partial.y;
            fp_acc[2] += partial.z;
            fp_acc[3] += partial.w;
        }

        if (full_cols)
        {
            const float4 scale = *reinterpret_cast<const float4*>(d_scale_B + n_base);
            if (kb_outer == 1)
            {
                d_C_fp32[n_base + 0] = fp_acc[0] * scale.x * alpha;
                d_C_fp32[n_base + 1] = fp_acc[1] * scale.y * alpha;
                d_C_fp32[n_base + 2] = fp_acc[2] * scale.z * alpha;
                d_C_fp32[n_base + 3] = fp_acc[3] * scale.w * alpha;
            }
            else
            {
                atomicAdd(&d_C_fp32[n_base + 0], fp_acc[0] * scale.x * alpha);
                atomicAdd(&d_C_fp32[n_base + 1], fp_acc[1] * scale.y * alpha);
                atomicAdd(&d_C_fp32[n_base + 2], fp_acc[2] * scale.z * alpha);
                atomicAdd(&d_C_fp32[n_base + 3], fp_acc[3] * scale.w * alpha);
            }
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const float partial = fp_acc[c] * d_scale_B[n] * alpha;
                    if (kb_outer == 1)
                    {
                        d_C_fp32[n] = partial;
                    }
                    else
                    {
                        atomicAdd(&d_C_fp32[n], partial);
                    }
                }
            }
        }
    }
}


// ============================================================================
// Q/Wo 2x-unrolled LDS K-reduction blockwise kernel — TN=128, CPT=2 variant
//
// Same ILP strategy as the TN=256 version but for the TN=128/CPT=2 tile.
// Uses int2 vectorized loads (8 bytes) like the standard TN=128 kernel,
// but processes 2 activation blocks simultaneously with separate accumulators.
// Beneficial when WK is small enough that each wave gets 2+ blocks.
// ============================================================================
template <int WK, int ACT_BK = 32>
__global__ __launch_bounds__(64 * WK)
void gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    const float* __restrict__ d_scale_B,
    int N,
    int K,
    int kb_outer,
    float alpha)
{
    static_assert(WK >= 2 && WK <= 16, "WK must be 2-16");
    static_assert(ACT_BK == 32 || ACT_BK == 64 || ACT_BK == 128, "ACT_BK must be 32, 64, or 128");
    constexpr int KGROUPS_PER_ACT = ACT_BK / 4;

    constexpr int TILE_N = 128;
    constexpr int CPT = 2;
    constexpr int THREADS_PER_WAVE = TILE_N / CPT;
    static_assert(THREADS_PER_WAVE == 64, "TILE_N/CPT must equal 64 (one wavefront)");

    const int wave_id = threadIdx.x / THREADS_PER_WAVE;
    const int lane_id = threadIdx.x % THREADS_PER_WAVE;
    const int n_base = blockIdx.x * TILE_N + lane_id * CPT;
    const int outer_block = blockIdx.y;
    const int act_blocks = K / ACT_BK;

    const int act_blocks_per_outer = (act_blocks + kb_outer - 1) / kb_outer;
    const int outer_act_start = outer_block * act_blocks_per_outer;
    const int outer_act_end = min(outer_act_start + act_blocks_per_outer, act_blocks);
    if (outer_act_start >= act_blocks)
    {
        return;
    }

    const int outer_act_count = outer_act_end - outer_act_start;
    const int act_per_wave_base = outer_act_count / WK;
    const int act_remainder = outer_act_count % WK;
    const int my_act_start = outer_act_start + wave_id * act_per_wave_base + min(wave_id, act_remainder);
    const int my_act_count = act_per_wave_base + (wave_id < act_remainder ? 1 : 0);
    const int my_act_end = my_act_start + my_act_count;

    float fp_acc[2];
    fp_acc[0] = 0.0f;
    fp_acc[1] = 0.0f;

    if (n_base < N && my_act_count > 0)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        // Process pairs of activation blocks for ILP
        int act_block = my_act_start;
        const int pair_end = my_act_start + (my_act_count & ~1);

        #pragma unroll 1
        for (; act_block < pair_end; act_block += 2)
        {
            int32_t acc0[2] = {0, 0};
            int32_t acc1[2] = {0, 0};
            const int kg_base0 = act_block * KGROUPS_PER_ACT;
            const int kg_base1 = (act_block + 1) * KGROUPS_PER_ACT;

            #pragma unroll
            for (int kgi = 0; kgi < KGROUPS_PER_ACT; ++kgi)
            {
                const int32_t a0 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base0 + kgi) * 4);
                const int2 b0 = *reinterpret_cast<const int2*>(b_row + (kg_base0 + kgi) * b_stride);
                acc0[0] = __builtin_amdgcn_sdot4(a0, b0.x, acc0[0], false);
                acc0[1] = __builtin_amdgcn_sdot4(a0, b0.y, acc0[1], false);

                const int32_t a1 = *reinterpret_cast<const int32_t*>(d_A_int8 + (kg_base1 + kgi) * 4);
                const int2 b1 = *reinterpret_cast<const int2*>(b_row + (kg_base1 + kgi) * b_stride);
                acc1[0] = __builtin_amdgcn_sdot4(a1, b1.x, acc1[0], false);
                acc1[1] = __builtin_amdgcn_sdot4(a1, b1.y, acc1[1], false);
            }

            const float scale0 = d_scales_A_blockwise[act_block];
            const float scale1 = d_scales_A_blockwise[act_block + 1];
            fp_acc[0] += static_cast<float>(acc0[0]) * scale0;
            fp_acc[1] += static_cast<float>(acc0[1]) * scale0;
            fp_acc[0] += static_cast<float>(acc1[0]) * scale1;
            fp_acc[1] += static_cast<float>(acc1[1]) * scale1;
        }

        // Handle odd tail
        if (act_block < my_act_end)
        {
            int32_t block_acc[2] = {0, 0};
            gemv_int8_accumulate_act_block<2, KGROUPS_PER_ACT>(block_acc, d_A_int8, b_row, b_stride,
                                           act_block * KGROUPS_PER_ACT);
            gemv_int8_flush_blockwise_acc(block_acc, fp_acc, d_scales_A_blockwise, act_block);
        }
    }

    // LDS K-reduction
    __shared__ float s_partial[WK - 1][TILE_N];

    if (wave_id > 0 && n_base < N)
    {
        s_partial[wave_id - 1][lane_id * CPT + 0] = fp_acc[0];
        s_partial[wave_id - 1][lane_id * CPT + 1] = fp_acc[1];
    }

    __syncthreads();

    if (wave_id == 0 && n_base < N)
    {
        #pragma unroll
        for (int w = 0; w < WK - 1; ++w)
        {
            fp_acc[0] += s_partial[w][lane_id * CPT + 0];
            fp_acc[1] += s_partial[w][lane_id * CPT + 1];
        }

        const float partial0 = fp_acc[0] * d_scale_B[n_base + 0] * alpha;
        const float partial1 = fp_acc[1] * d_scale_B[n_base + 1] * alpha;
        if (kb_outer == 1)
        {
            float2 out_val = make_float2(partial0, partial1);
            *reinterpret_cast<int2*>(&d_C_fp32[n_base]) =
                *reinterpret_cast<const int2*>(&out_val);
        }
        else
        {
            atomicAdd(&d_C_fp32[n_base + 0], partial0);
            atomicAdd(&d_C_fp32[n_base + 1], partial1);
        }
    }
}


// ============================================================================
// Fused FP32→INT8 GEMV kernel: Quantize + GEMV + Scale in one launch
//
// Eliminates the 3-kernel pipeline (quantize → GEMV → scale) by:
//   1. Loading FP32 activations to shared memory per K-block
//   2. Computing block-local absmax for INT8 quantization
//   3. Quantizing in-register and using v_dot4_i32_i8 for the dot product
//   4. Converting INT32 accumulator to FP32 with (block_scale * scale_B[n])
//   5. atomicAdd FP32 partial results across K-blocks
//   6. Optional fused bias addition
//
// Memory traffic: Same weight reads as INT8 GEMV, but no separate
// activation INT8 buffer, no INT32 output buffer, no scale_A buffer.
// Activations (14KB for K=3584) stay in L2 after first projection.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N/CPT)
// Same grid structure as grid_kpar, but FP32 in → FP32 out.
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_fp32_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_C_fp32,                // [N] FP32 output (zeroed before launch)
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int K, int kblocks)
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    // Process K in chunks of QUANT_TILE_K elements at a time for quantization.
    // Each chunk: load FP32 → find absmax → quantize → dot product.
    // 256 elements = 1KB FP32 in shared memory, fits easily.
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    // Each block processes its K-segment: [k_start, k_end)
    // Round k_per UP to a multiple of 4 so every k_start aligns with
    // VNNI K-groups (4 consecutive K values packed together).
    // Without this, k_start for intermediate blocks can land mid-group,
    // causing activation/weight misalignment in sdot4.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // After rounding k_per up, the last block(s) may have k_start >= K.
    // Those blocks have nothing to do — skip entirely.
    if (k_start >= K) return;

    // Shared memory: FP32 activations tile + reduction scratch
    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;        // Block-wide absmax for this tile
    __shared__ int8_t s_A_int8[QUANT_TILE_K]; // Quantized activations

    // FP32 accumulators for partial results (will be atomicAdd'd)
    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    // Process K-segment in QUANT_TILE_K-sized chunks
    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // --- Phase 1: Cooperative load of FP32 activations to shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        }
        __syncthreads();

        // --- Phase 2: Find absmax via parallel reduction ---
        // Each thread finds local max over its portion
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        // Warp reduction (wavefront = 64 threads on gfx906)
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Broadcast absmax from lane 0 to all lanes via __shfl (no LDS, no sync).
        // (s_absmax LDS slot is reused later in Phase 6 as a completion flag.)
        const float absmax = __shfl(local_max, 0);
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // --- Phase 3: Cooperative quantize FP32 → INT8 in shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // --- Phase 4: INT8 dot product using v_dot4_i32_i8 ---
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, tile_acc[0], false);
                    tile_acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, tile_acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, tile_acc[c], false);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
            }
        }

        __syncthreads();  // Ensure shared memory is safe for next tile
    }

    // --- Phase 5: Apply weight scales, write to output ---
    if (active)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            float result = f_acc[c] * d_scales_B[n_base + c];
            if (kblocks == 1)
                d_C_fp32[n_base + c] = result;      // Plain store — no contention
            else
                atomicAdd(&d_C_fp32[n_base + c], result);
        }
    }

    // Bias: only apply once across all K-blocks (last block does it)
    // We use k_block == kblocks - 1 as the "last" block.
    // Note: this has a data race if atomicAdds from other blocks haven't
    // completed yet. Instead, bias is applied as a separate follow-up.
    // See dispatch function for bias handling.
}


// ============================================================================
// Fused FP32→INT8 GEMV scatter kernel: Quantize + GEMV → partial buffer
//
// Same inner loop as gemv_fused_fp32_int8_vnni_kernel_t but writes unscaled
// FP32 partials to d_partial[k_block * N + n], enabling grid-level K-splitting
// WITHOUT atomicAdd or memset. Paired with gemv_reduce_scale_kernel_t to
// produce the final FP32 output.
//
// This replaces the 3-kernel pipeline (quantize → GEMV → scale) with a
// 2-kernel pipeline (fused_scatter → reduce+scale) that eliminates:
//   - The separate quantize kernel launch
//   - hipMemsetAsync for atomicAdd zeroing
//   - All atomicAdd contention
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_scatter_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,           // [KB × N] FP32 partial buffer
    int N, int K, int kblocks)
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;
    __shared__ int8_t s_A_int8[QUANT_TILE_K];

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // Phase 1: Cooperative load of FP32 activations to shared memory
        for (int i = tid; i < tile_len; i += THREADS)
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        __syncthreads();

        // Phase 2: Find absmax via warp reduction
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Broadcast absmax from lane 0 to all lanes via __shfl (no LDS, no sync).
        const float absmax = __shfl(local_max, 0);
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // Phase 3: Cooperative quantize FP32 → INT8 in shared memory
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // Phase 4: INT8 dot product using v_dot4_i32_i8
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, tile_acc[0], false);
                    tile_acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, tile_acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, tile_acc[c], false);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
        }

        __syncthreads();
    }

    // Phase 5: Write unscaled FP32 partials to buffer (plain store, no atomicAdd)
    if (active)
    {
        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = f_acc[c];
    }
}


// ============================================================================
// Reduce + scale kernel: Sum KB FP32 partials, apply scale_B + bias
//
// Used after gemv_fused_scatter_int8_vnni_kernel_t to combine K-block
// partial results into the final FP32 output with scaling and bias.
//
// Grid: dim3((N + TILE_N - 1) / TILE_N), Block: dim3(TILE_N / CPT)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_reduce_scale_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB × N] partial buffer
    float* __restrict__ d_C_fp32,                // [N] output
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int KB,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N) return;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < KB; ++kb)
            sum += d_partial_fp32[static_cast<size_t>(kb) * N + n];

        float result = sum * d_scales_B[n] * alpha;
        if (beta != 0.0f && d_C_existing)
            result += beta * d_C_existing[n];
        if (d_bias)
            result += d_bias[n];

        d_C_fp32[n] = result;
    }
}


// ============================================================================
// Self-reducing scatter kernel: fused scatter + inline reduce (single kernel)
//
// Instead of writing partials to a buffer and launching a separate reduce
// kernel, the last k-block to complete each n-block sums all partials inline
// using a "last-to-arrive" atomic counter pattern.
//
// Benefits:
//   - Eliminates reduce kernel launch overhead (5-27μs savings per shape)
//   - Single kernel launch instead of two
//   - INT32 atomicAdd for counter (native HW on gfx906, ~1 cycle)
//
// Requires: d_counter array must be zero-initialized before launch.
//           d_counter has ceil(N/TILE_N) elements.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_scatter_selfreduce_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,           // [KB × N] FP32 partial buffer
    float* __restrict__ d_C_fp32,                // [N] FP32 output
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int* __restrict__ d_counter,                 // [grid_n] atomic completion counters
    int N, int K, int kblocks, int valid_kb,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;
    __shared__ int8_t s_A_int8[QUANT_TILE_K];

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // Phase 1: Cooperative load of FP32 activations to shared memory
        for (int i = tid; i < tile_len; i += THREADS)
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        __syncthreads();

        // Phase 2: Find absmax via warp reduction
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Broadcast absmax from lane 0 to all lanes via __shfl (no LDS, no sync).
        // (s_absmax LDS slot is reused later in Phase 6 as a completion flag.)
        const float absmax = __shfl(local_max, 0);
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // Phase 3: Cooperative quantize FP32 → INT8 in shared memory
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // Phase 4: INT8 dot product using v_dot4_i32_i8
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, tile_acc[0], false);
                    tile_acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, tile_acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, tile_acc[c], false);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
        }
    }

    // Phase 5: Write FP32 partials to buffer (plain store, no atomicAdd)
    if (active)
    {
        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = f_acc[c];
    }

    // Phase 6: Last-to-arrive reduction
    // Make partial writes globally visible, then atomically increment counter.
    // The last k-block to arrive for each n-block sums all partials and
    // writes the final output (eliminating the separate reduce kernel).
    __threadfence();

    if (tid == 0)
    {
        // Shared flag: did this block's lane 0 observe completion?
        // We use s_absmax (reusing LDS since scatter phase is done)
        unsigned int old = atomicAdd(&d_counter[blockIdx.x], 1);
        // Store 1.0 if last-to-arrive, 0.0 otherwise (broadcast via LDS)
        s_absmax = (old + 1 == static_cast<unsigned int>(valid_kb)) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Only the last k-block's wavefront performs the reduction
    if (s_absmax > 0.0f && active)
    {
        // Sum all valid_kb partials for this thread's columns
        float sum[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) sum[c] = 0.0f;

        for (int kb = 0; kb < valid_kb; ++kb)
        {
            const size_t off = static_cast<size_t>(kb) * N + n_base;
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                sum[c] += d_partial_fp32[off + c];
        }

        // Apply scale_B, alpha, beta, bias and write final output
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            float result = sum[c] * d_scales_B[n] * alpha;
            if (beta != 0.0f && d_C_existing)
                result += beta * d_C_existing[n];
            if (d_bias)
                result += d_bias[n];
            d_C_fp32[n] = result;
        }
    }
}


// ============================================================================
// INT8-input DIRECT kernel: pre-quantized GEMV → final FP32 output
//
// Single-kernel path for kb=1 (no K-parallelism). Instead of writing to a
// partial buffer and launching a separate reduce, this kernel writes the
// final scaled+biased FP32 output directly. Eliminates the reduce kernel
// entirely, saving ~2ms per call on large-N projections (e.g., LM_HEAD).
//
// Grid: dim3(grid_n), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_direct_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,        // [K] pre-quantized INT8 activations
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] VNNI weights
    float* __restrict__ d_C_fp32,               // [N] FP32 output (final)
    const float* __restrict__ d_scale_A,         // [1] activation scale or [K/32] blockwise scales
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int K,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            #pragma unroll 1
            for (int act_block = 0; act_block < act_blocks; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int k_groups = K / 4;

            #pragma unroll 4
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            float result = fp_acc[c] * d_scales_B[n] * alpha;
            if (beta != 0.0f && d_C_existing)
                result += beta * d_C_existing[n];
            if (d_bias)
                result += d_bias[n];
            d_C_fp32[n] = result;
        }
    }
}


// ============================================================================
// INT8-input scatter kernel: pre-quantized GEMV → partial buffer
//
// Like gemv_fused_scatter_int8_vnni_kernel_t but takes pre-quantized INT8
// activations instead of FP32. This enables "quantize once, compute many"
// for multiply_fused_tensor (Q/K/V/Wo/Gate/Up/Down share one quantize).
//
// Key advantages over FP32 scatter:
//   1. No shared memory → 0 bytes LDS → 10 waves/SIMD (max occupancy)
//   2. No __syncthreads() → no serialization overhead
//   3. 1 byte/element activation reads vs 4 bytes → 4× less activation BW
//   4. No redundant per-projection quantization
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_scatter_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,        // [K] pre-quantized INT8 activations
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,          // [KB × N] FP32 partial buffer
    const float* __restrict__ d_scale_A,         // [1] activation scale or [K/32] blockwise scales
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int act_blocks_per_raw = (act_blocks + kblocks - 1) / kblocks;
            const int act_block_start = k_block * act_blocks_per_raw;
            const int act_block_end = min(act_block_start + act_blocks_per_raw, act_blocks);

            if (act_block_start >= act_blocks) return;

            #pragma unroll 1
            for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;

            #pragma unroll 4
            for (int kg = kg_start; kg < kg_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = fp_acc[c];
    }
}


// ============================================================================
// INT8-input self-reducing scatter kernel: pre-quantized GEMV with
// last-to-arrive inline reduction (single kernel, no reduce launch)
//
// Same inner loop as gemv_int8_scatter_vnni_kernel_t but the last k-block
// to complete each n-tile sums all partials and writes the final FP32 output.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_scatter_selfreduce_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,        // [K] pre-quantized INT8 activations
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,          // [KB × N] FP32 partial buffer
    float* __restrict__ d_C_fp32,               // [N] FP32 output
    const float* __restrict__ d_scale_A,         // [1] activation scale or [K/32] blockwise scales
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int* __restrict__ d_counter,                 // [grid_n] atomic completion counters
    int N, int K, int kblocks, int valid_kb,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int act_blocks_per_raw = (act_blocks + kblocks - 1) / kblocks;
            const int act_block_start = k_block * act_blocks_per_raw;
            const int act_block_end = min(act_block_start + act_blocks_per_raw, act_blocks);

            if (act_block_start >= act_blocks) return;

            #pragma unroll 1
            for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;

            #pragma unroll 4
            for (int kg = kg_start; kg < kg_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = fp_acc[c];
    }

    // Phase 2: Last-to-arrive reduction via atomic counter
    __threadfence();

    __shared__ float s_last_flag;  // Reuse to broadcast last-to-arrive status

    if (tid == 0)
    {
        unsigned int old = atomicAdd(&d_counter[blockIdx.x], 1);
        s_last_flag = (old + 1 == static_cast<unsigned int>(valid_kb)) ? 1.0f : 0.0f;
    }
    __syncthreads();

    if (s_last_flag > 0.0f && active)
    {
        // Sum all valid_kb partials
        float sum[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) sum[c] = 0.0f;

        for (int kb = 0; kb < valid_kb; ++kb)
        {
            const size_t off = static_cast<size_t>(kb) * N + n_base;
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                sum[c] += d_partial_fp32[off + c];
        }

        // Apply scale_B, alpha, beta, bias — write final output
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            float result = sum[c] * d_scales_B[n] * alpha;
            if (beta != 0.0f && d_C_existing)
                result += beta * d_C_existing[n];
            if (d_bias)
                result += d_bias[n];
            d_C_fp32[n] = result;
        }
    }
}


// ============================================================================
// Batched INT8-input scatter kernel: multiple projections in one launch
//
// Processes multiple projections (e.g., Q/K/V or Gate/Up) that share the
// same pre-quantized INT8 activations and K dimension. Each thread block
// determines which projection it belongs to via blockIdx.x → batch lookup.
//
// Partial buffer layout: concatenated N across batches.
//   partial[kb * total_N + n_offset[batch] + local_n]
//
// Benefits over individual launches:
//   1. 3→1 kernel launches for QKV (2→1 for Gate/Up)
//   2. Better GPU occupancy (K/V alone = 4 n-blocks = can't fill 60 CUs;
//      batched QKV = 36 n-blocks = much better utilization)
//   3. Single input activation read amortized across all projections
//
// Grid: dim3(total_grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_scatter_batched_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,    // [K] pre-quantized INT8 activations
    const float* __restrict__ d_scale_A,     // [1] activation scale or [K/32] blockwise scales
    float* __restrict__ d_partial_fp32,      // [KB × total_N] FP32 partial buffer
    GemvBatchArgs args,                      // Batch descriptors (by value)
    int K, int kblocks)
{
    // Phase 1: Determine which batch this block belongs to
    // Linear scan over prefix sums (max 8 entries, trivial cost)
    int batch = 0;
    #pragma unroll
    for (int b = 1; b < MAX_GEMV_BATCH; ++b)
    {
        if (b < args.num_batches && blockIdx.x >= args.grid_n_offset[b])
            batch = b;
    }

    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_local = local_block_x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_local < args.N[batch]);
    const int k_block = blockIdx.y;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        // Index into THIS batch's weight matrix using local n position
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(args.d_B[batch]) + n_local;
        const size_t b_stride = static_cast<size_t>(args.N[batch]);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int act_blocks_per_raw = (act_blocks + kblocks - 1) / kblocks;
            const int act_block_start = k_block * act_blocks_per_raw;
            const int act_block_end = min(act_block_start + act_blocks_per_raw, act_blocks);

            if (act_block_start >= act_blocks) return;

            #pragma unroll 1
            for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;

            #pragma unroll 4
            for (int kg = kg_start; kg < kg_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_pair.x, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_pair.y, acc[1], false);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[c], false);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        const int n_global = args.n_offset[batch] + n_local;
        const size_t partial_offset = static_cast<size_t>(k_block) * args.total_N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_global + c] = fp_acc[c];
    }
}


// ============================================================================
// Batched reduce+scale kernel: reduces partials for multiple projections
//
// Reads from concatenated partial buffer and writes to per-projection outputs
// with per-projection scale_B, bias, alpha/beta.
//
// Grid: dim3(total_grid_n), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_reduce_scale_batched_kernel_t(
    const float* __restrict__ d_partial_fp32,  // [KB × total_N] partial buffer
    GemvBatchArgs args,                         // Batch descriptors
    int KB,
    float alpha, float beta)
{
    // Find which batch this block belongs to
    int batch = 0;
    #pragma unroll
    for (int b = 1; b < MAX_GEMV_BATCH; ++b)
    {
        if (b < args.num_batches && blockIdx.x >= args.grid_n_offset[b])
            batch = b;
    }

    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_local = local_block_x * TILE_N + threadIdx.x * CPT;
    const int n_global = args.n_offset[batch] + n_local;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        if (n_local + c >= args.N[batch]) return;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < KB; ++kb)
            sum += d_partial_fp32[static_cast<size_t>(kb) * args.total_N + n_global + c];

        float result = sum * args.d_scales_B[batch][n_local + c] * alpha;
        if (args.d_bias[batch])
            result += args.d_bias[batch][n_local + c];

        args.d_C[batch][n_local + c] = result;
    }
}


// ============================================================================

// C API: Launch wrappers
// ============================================================================

extern "C" {

bool rocmGemv_int8_scatter_vnni_blockwise(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scales_A_blockwise,
    const float* d_scales_B,
    const float* d_bias,
    float* d_partial_buf,
    int* d_counter,
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,
    int device_id, void* stream);


void rocmGemv_int8_vnni_set_tuning_overrides(
    int tn,
    int kb)
{
    g_int8_vnni_tn_override.store(tn, std::memory_order_relaxed);
    g_int8_vnni_kb_override.store(kb, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_tuning_overrides()
{
    rocmGemv_int8_vnni_set_tuning_overrides(-1, -1);
}

void rocmGemv_int8_vnni_set_wk_override(int wk)
{
    g_int8_vnni_wk_override.store(wk, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_unroll_override(int unroll)
{
    g_int8_vnni_unroll_override.store(unroll, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_act_block_override(int act_bk)
{
    g_int8_vnni_act_block_override.store(act_bk, std::memory_order_relaxed);
}

int rocmGemv_int8_vnni_get_act_block_k()
{
    int v = g_int8_vnni_act_block_override.load(std::memory_order_relaxed);
    return (v == 64 || v == 128) ? v : 32;
}

void rocmGemv_int8_vnni_reset_qwo_overrides()
{
    g_int8_vnni_wk_override.store(-1, std::memory_order_relaxed);
    g_int8_vnni_unroll_override.store(-1, std::memory_order_relaxed);
    g_int8_vnni_act_block_override.store(-1, std::memory_order_relaxed);
    g_int8_vnni_skip_memset_override.store(0, std::memory_order_relaxed);
    g_int8_vnni_force_pair_override.store(0, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_skip_memset(int skip)
{
    g_int8_vnni_skip_memset_override.store(skip, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_force_pair(int force)
{
    g_int8_vnni_force_pair_override.store(force, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_set_wide_tuning_overrides(
    int wide_tn,
    int wide_cpt,
    int wide_vec_load)
{
    g_int8_vnni_wide_tn_override.store(wide_tn, std::memory_order_relaxed);
    g_int8_vnni_wide_cpt_override.store(wide_cpt, std::memory_order_relaxed);
    g_int8_vnni_wide_vec_load_override.store(wide_vec_load, std::memory_order_relaxed);
}

void rocmGemv_int8_vnni_reset_wide_tuning_overrides()
{
    rocmGemv_int8_vnni_set_wide_tuning_overrides(-1, -1, -1);
}

bool rocmGemv_int8_int8_int32_vnni(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    // Grid-level K-parallel: handles ALL non-wide_vec4 shapes with N%4==0.
    // Previous wide2 path (N>=3K, N<8K) only generated N/256 blocks → 148 waves
    // for FFN Gate/Up, achieving just 17% peak BW. Grid_kpar K-splits via
    // blockIdx.y to generate thousands of blocks, fully saturating the GPU.
    // Profiling: grid_kpar achieves 488-649 GB/s (48-63% peak) vs wide2's 176 GB/s.
    const bool use_grid_kpar = !use_wide_vec4 && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);
    const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
                }
            }
        }
    }
    else if (use_grid_kpar)
    {
        // Grid-level K-parallel with atomicAdd: KB independent blocks per output tile
        // Each block processes K/KB fraction; output must be zeroed before launch
        //
        // KB heuristic: unified formula for all shapes.
        //
        // Key insight from exhaustive TN×KB sweep (MI50 gfx906, Qwen7B shapes):
        //   ~8 waves/CU is the sweet spot for ALL shapes. More waves just adds
        //   atomicAdd contention per N-tile without improving BW utilization.
        //   Additionally, KB values that evenly divide k_groups avoid tail-effect
        //   work imbalance (some wavefronts processing fewer k-groups than others).
        //
        // Sweep results (optimal KB per shape):
        //   Q/Wo (3584×3584):     KB=16, 7.5 w/CU, 56 kgrp/wave → 456 GB/s (46%)
        //   K/V (512×3584):       KB=56, 3.7 w/CU, 16 kgrp/wave → 129 GB/s (13%)
        //   FFN Gate/Up (18944×3584): KB=4, 9.9 w/CU, 224 kgrp/wave → 756 GB/s (76%)
        //   FFN Down (3584×18944):    KB=16, 7.5 w/CU, 296 kgrp/wave → 760 GB/s (76%)
        //
        // Strategy:
        //   1. Compute min KB for ~8 waves/CU (memory latency hiding)
        //   2. Cap so inner loop runs ≥16 k-groups (proven floor from sweep)
        //   3. Round to nearest factor of k_groups (avoid uneven work distribution)
        //      Bias toward lower factor on ties (fewer atomicAdds = less contention)
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
        constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE_TILE_N;  // 256
        constexpr int NUM_CUS = 60;  // MI50

        int tn = TN;
        if (int8_tn_override == TN_WIDE)
        {
            tn = TN_WIDE;
        }
        else if (int8_tn_override == TN)
        {
            tn = TN;
        }

        const int grid_n = (N + tn - 1) / tn;
        const int k_groups = K / 4;

        constexpr int TARGET_WAVES_PER_CU = 8;
        constexpr int MIN_KGROUPS_PER_WAVE = 16;  // proven floor from sweep

        int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
        int kb_raw = max(2, (target_blocks + grid_n - 1) / grid_n);

        // Cap: ensure minimum inner loop length
        int kb_max = max(2, k_groups / MIN_KGROUPS_PER_WAVE);
        kb_raw = min(kb_raw, kb_max);

        // Round to nearest factor of k_groups (prefer lower on tie → fewer atomicAdds)
        int kb = kb_raw;
        if (k_groups % kb_raw != 0)
        {
            for (int d = 1; d < kb_raw; ++d)
            {
                if (kb_raw - d >= 2 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
                if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
            }
        }

        if (int8_kb_override > 0)
        {
            kb = int8_kb_override;
        }

        (void)hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), static_cast<hipStream_t>(stream));  // NOLINT(clang-diagnostic-unused-result)

        if (tn == TN_WIDE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN_WIDE, CPT>),
                dim3(grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
        }
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// Pair dispatch for tiny K/V shapes: launches the LDS K-reduce pair kernel
// for two projections with same K, each with N <= 512 and K >= 2048.
// ---------------------------------------------------------------------------
bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
    const int8_t* d_A_int8,
    const float* d_scales_A_blockwise,
    const int8_t* d_B0,
    const int8_t* d_B1,
    float* d_C0,
    float* d_C1,
    const float* d_scales_B0,
    const float* d_scales_B1,
    int N0, int N1,
    int K,
    float alpha,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int WK = 8;

    const int grid_n0 = (N0 + TN - 1) / TN;
    const int grid_n1 = (N1 + TN - 1) / TN;
    const int total_grid_n = grid_n0 + grid_n1;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);
    // Phase 3: Use unified KB selector (two-regime + occupancy floor + KB cap)
    // instead of legacy tiny-KV selector. Aligns pair dispatch with Phase 1
    // individual dispatch tuning.
    const int kb_outer = select_blockwise_qwo_outer_splits(total_grid_n, act_blocks, int8_kb_override, WK);

    // Zero outputs when KB > 1 (kernel uses atomicAdd)
    if (kb_outer > 1)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_init_output_pair_kernel_t<TN, CPT>),
            dim3(total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_C0, d_C1, nullptr, nullptr,
            N0, N1, grid_n0, 0.0f);
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_int8_int8_lds_kreduce_pair_blockwise_scaled_kernel_t<TN, CPT, WK>),
        dim3(total_grid_n, kb_outer), dim3((TN / CPT) * WK), 0, static_cast<hipStream_t>(stream),
        d_A_int8,
        d_B0, d_B1,
        d_C0, d_C1,
        d_scales_A_blockwise,
        d_scales_B0, d_scales_B1,
        N0, N1, grid_n0,
        K, kb_outer, alpha);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair] launch failed: %s (N0=%d N1=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N0, N1, K, kb_outer);
        return false;
    }

    return true;
}

bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scales_A_blockwise,
    const float* d_scale_B,
    int N, int K,
    float alpha,
    float beta,
    const float* d_C_existing,
    const float* d_bias,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }
    // Check K is divisible by the current act_block size (32, 64, or 128)
    const int effective_act_bk = rocmGemv_int8_vnni_get_act_block_k();
    if ((K % effective_act_bk) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] K must be multiple of act_block_k=%d (N=%d K=%d)\n",
                effective_act_bk, N, K);
        return false;
    }
    if (beta != 0.0f && d_C_existing == nullptr)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] d_C_existing required when beta != 0 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_vec4 = ((N % 4) == 0);
    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    const bool use_wide_vec4 = (N >= 8 * K) && (N >= wide_tn);
    const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);
    const int qwo_tn = (int8_tn_override == 128 || int8_tn_override == 256) ? int8_tn_override : 128;

    // Unified LDS k-reduce path: routes all vec4-aligned shapes through the
    // high-performance LDS k-reduce kernel. The kernel has N bounds checking
    // (n_base < N guard on all reads/writes), so it handles N < TN=128 correctly
    // with grid_n=1 — excess threads simply skip. Previously gated on N >= 128,
    // but that caused TP-sharded K/V projections (N=32-64) to fall through to the
    // slow fallback kernel (TILE_N=256, no KB splitting, 2-4x slower).
    // The occupancy selector adapts KB to fill CUs at any grid_n.
    const bool use_lds_kreduce_blockwise = !use_wide_vec4
        && use_vec4;  // Any vec4-aligned shape (N % 4 == 0)
    const bool use_grid_kpar = !use_wide_vec4 && !use_lds_kreduce_blockwise && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = !use_wide_vec4 && !use_grid_kpar && (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);

    if (use_lds_kreduce_blockwise)
    {
        // Unified Q/Wo dispatch with WK, unroll, and act_block overrides
        const int wk_ov = g_int8_vnni_wk_override.load(std::memory_order_relaxed);
        const int unroll_ov = g_int8_vnni_unroll_override.load(std::memory_order_relaxed);
        const int tn = qwo_tn;  // 128 or 256
        const int grid_n = (N + tn - 1) / tn;

        // WK=8 default (tuned). WK=16 infrastructure available via override for future
        // tuning — static_asserts and switch cases support WK up to 16, but auto-select
        // stays at 8 since atomicAdd is not the bottleneck for small-grid_n shapes
        // (latency floor dominates; total wave count is unchanged since WK×KB = const).
        const int wk = (wk_ov >= 2 && wk_ov <= 16) ? wk_ov : 8;
        const bool use_unroll2 = (unroll_ov == -1) ? true : (unroll_ov == 2);  // default: unroll=2 (tuned)
        const int act_bk = rocmGemv_int8_vnni_get_act_block_k();  // 32, 64, or 128

        const int act_blocks = K / act_bk;
        const int kb_outer = select_blockwise_qwo_outer_splits(grid_n, act_blocks, int8_kb_override, wk);

        if (kb_outer > 1 && beta == 0.0f && d_bias == nullptr)
        {
            (void)hipMemsetAsync(d_C_fp32, 0, static_cast<size_t>(N) * sizeof(float), static_cast<hipStream_t>(stream));
        }
        else if (beta != 0.0f || d_bias != nullptr)
        {
            if (tn == 256)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<256, 4>),
                    dim3(grid_n), dim3(64), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
            else
            {
                constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
                constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<TN, CPT>),
                    dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
        }

        // Macro to dispatch with runtime WK -> compile-time template arg
        #define LAUNCH_QWO_KERNEL(KERNEL_TEMPLATE, WK_VAL)                                        \
            hipLaunchKernelGGL(                                                                    \
                HIP_KERNEL_NAME(KERNEL_TEMPLATE<WK_VAL>),                                         \
                dim3(grid_n, kb_outer), dim3(64 * WK_VAL), 0, static_cast<hipStream_t>(stream),   \
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,               \
                N, K, kb_outer, alpha)

        #define LAUNCH_QWO_KERNEL_ABK(KERNEL_TEMPLATE, WK_VAL, ABK_VAL)                            \
            hipLaunchKernelGGL(                                                                    \
                HIP_KERNEL_NAME(KERNEL_TEMPLATE<WK_VAL, ABK_VAL>),                                \
                dim3(grid_n, kb_outer), dim3(64 * WK_VAL), 0, static_cast<hipStream_t>(stream),   \
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,               \
                N, K, kb_outer, alpha)

        #define LAUNCH_LDS_KERNEL(TN_VAL, CPT_VAL, WK_VAL)                                       \
            hipLaunchKernelGGL(                                                                    \
                HIP_KERNEL_NAME(gemv_int8_int8_lds_kreduce_blockwise_scaled_kernel_t<TN_VAL, CPT_VAL, WK_VAL>), \
                dim3(grid_n, kb_outer), dim3(64 * WK_VAL), 0, static_cast<hipStream_t>(stream),   \
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,               \
                N, K, kb_outer, alpha)

        if (tn == 256 && use_unroll2)
        {
            switch (wk)
            {
            case 2: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 2); break;
            case 3: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 3); break;
            case 6: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 6); break;
            case 8: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 8); break;
            case 16: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 16); break;
            default: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_lds_kreduce_blockwise_scaled_kernel_t, 4); break;
            }
        }
        else if (tn == 256)
        {
            switch (wk)
            {
            case 2: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 2); break;
            case 3: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 3); break;
            case 6: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 6); break;
            case 8: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 8); break;
            case 16: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 16); break;
            default: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_lds_kreduce_blockwise_scaled_kernel_t, 4); break;
            }
        }
        else if (tn == 128 && use_unroll2)
        {
            // TN=128, CPT=2, unroll2 variant — dispatch with act_block size
            if (act_bk == 128)
            {
                switch (wk)
                {
                case 2: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 2, 128); break;
                case 3: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 3, 128); break;
                case 6: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 6, 128); break;
                case 8: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 8, 128); break;
                case 16: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 16, 128); break;
                default: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 4, 128); break;
                }
            }
            else if (act_bk == 64)
            {
                switch (wk)
                {
                case 2: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 2, 64); break;
                case 3: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 3, 64); break;
                case 6: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 6, 64); break;
                case 8: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 8, 64); break;
                case 16: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 16, 64); break;
                default: LAUNCH_QWO_KERNEL_ABK(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 4, 64); break;
                }
            }
            else
            {
                switch (wk)
                {
                case 2: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 2); break;
                case 3: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 3); break;
                case 6: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 6); break;
                case 8: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 8); break;
                case 16: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 16); break;
                default: LAUNCH_QWO_KERNEL(gemv_int8_int8_qwo_unroll2_128_lds_kreduce_blockwise_scaled_kernel_t, 4); break;
                }
            }
        }
        else
        {
            // TN=128, CPT=2 standard
            constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
            constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
            switch (wk)
            {
            case 2: LAUNCH_LDS_KERNEL(TN, CPT, 2); break;
            case 3: LAUNCH_LDS_KERNEL(TN, CPT, 3); break;
            case 6: LAUNCH_LDS_KERNEL(TN, CPT, 6); break;
            case 8: LAUNCH_LDS_KERNEL(TN, CPT, 8); break;
            case 16: LAUNCH_LDS_KERNEL(TN, CPT, 16); break;
            default: LAUNCH_LDS_KERNEL(TN, CPT, 4); break;
            }
        }

        #undef LAUNCH_QWO_KERNEL
        #undef LAUNCH_QWO_KERNEL_ABK
        #undef LAUNCH_LDS_KERNEL

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] qwo dispatch failed: %s (N=%d K=%d tn=%d wk=%d unroll=%d act_bk=%d kb=%d)\n",
                    hipGetErrorString(err), N, K, tn, wk, use_unroll2 ? 2 : 1, act_bk, kb_outer);
            return false;
        }

        return true;
    }

    if (use_grid_kpar)
    {
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
        constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE2_TILE_N;

        int tn = TN;
        if (int8_tn_override == TN_WIDE)
        {
            tn = TN_WIDE;
        }
        else if (int8_tn_override == TN)
        {
            tn = TN;
        }

        const int grid_n = (N + tn - 1) / tn;
        const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

        // For tiny-grid decode shapes, the direct/scatter family is structurally better
        // than atomic grid_kpar because it avoids repeated output updates and can reduce
        // once per tile. Returning false asks the higher-level GEMM wrapper to use
        // its explicit workspace-backed scatter path instead of allocating scratch
        // inside this HIP translation unit.
        if (grid_n <= 8)
        {
            return false;
        }

        const int kb = select_blockwise_split_count(grid_n, act_blocks, int8_kb_override);

        if (kb > 1 && beta == 0.0f && d_bias == nullptr)
        {
            (void)hipMemsetAsync(d_C_fp32, 0, static_cast<size_t>(N) * sizeof(float), static_cast<hipStream_t>(stream));
        }
        else if (beta != 0.0f || d_bias != nullptr)
        {
            if (tn == TN_WIDE)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<TN_WIDE, CPT>),
                    dim3(grid_n), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_kernel_t<TN, CPT>),
                    dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    d_C_fp32, d_C_existing, d_bias, N, beta);
            }
        }

        if (tn == TN_WIDE)
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_blockwise_scaled_kernel_t<TN_WIDE, CPT>),
                dim3(grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, kb, alpha);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_blockwise_scaled_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, kb, alpha);
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] grid_kpar launch failed: %s (N=%d K=%d kb=%d)\n",
                    hipGetErrorString(err), N, K, kb);
            return false;
        }

        return true;
    }

    if (use_wide_vec4)
    {
        const int grid_n = (N + wide_tn - 1) / wide_tn;
        dim3 block(wide_tn / wide_cpt, 1, 1);

        if (wide_tn == 128)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<128, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else if (wide_tn == 512)
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<512, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
        else
        {
            if (wide_cpt == 2)
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 2, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 2, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
            else
            {
                if (wide_vec_load)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 4, true>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_vnni_blockwise_scaled_kernel_t<256, 4, false>), dim3(grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B, N, K, alpha, beta, d_C_existing, d_bias);
                }
            }
        }
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_blockwise_scaled_kernel,
            dim3(grid_n), block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_blockwise_scaled_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, d_scale_B,
            N, K, alpha, beta, d_C_existing, d_bias);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

bool rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched(
    const int8_t* d_A_int8,
    const float* d_scales_A_blockwise,
    int num_projections,
    const int8_t* const* d_B_ptrs,
    float* const* d_C_ptrs,
    const float* const* d_scales_B_ptrs,
    const float* const* d_bias_ptrs,
    const int* N_per_proj,
    int K,
    float alpha,
    float beta,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (num_projections < 1 || num_projections > MAX_GEMV_BATCH)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched] Invalid num_projections=%d (max=%d)\n",
                num_projections, MAX_GEMV_BATCH);
        return false;
    }
    if ((K % 4) != 0 || (K % GEMV_INT8_VNNI_ACT_BLOCK_K) != 0)
    {
        return false;
    }

    const int wide_tn_override = g_int8_vnni_wide_tn_override.load(std::memory_order_relaxed);
    const int wide_cpt_override = g_int8_vnni_wide_cpt_override.load(std::memory_order_relaxed);
    const int wide_vec_load_override = g_int8_vnni_wide_vec_load_override.load(std::memory_order_relaxed);

    int wide_tn = GEMV_INT8_VNNI_WIDE_TILE_N;
    if (wide_tn_override == 128 || wide_tn_override == 256 || wide_tn_override == 512)
    {
        wide_tn = wide_tn_override;
    }

    int wide_cpt = GEMV_INT8_VNNI_WIDE_DEFAULT_CPT;
    if (wide_cpt_override == 2 || wide_cpt_override == 4)
    {
        wide_cpt = wide_cpt_override;
    }

    bool wide_vec_load = true;
    if (wide_vec_load_override == 0)
    {
        wide_vec_load = false;
    }
    else if (wide_vec_load_override == 1)
    {
        wide_vec_load = true;
    }

    enum class GemvBatchKernelFamily
    {
        Wide,
        Square,
        Generic
    };

    GemvBatchKernelFamily families[MAX_GEMV_BATCH] = {};
    bool force_generic_pair_member[MAX_GEMV_BATCH] = {};

    const bool prefer_generic_pair_path =
        (num_projections == 2) &&
        (N_per_proj[0] == N_per_proj[1]) &&
        (N_per_proj[0] <= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N);

    for (int b = 0; b + 1 < num_projections; ++b)
    {
        const bool pair_same_n = (N_per_proj[b] == N_per_proj[b + 1]);
        const bool pair_small_n = (N_per_proj[b] <= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N);
        const bool pair_use_wide = (N_per_proj[b] >= 8 * K) && (N_per_proj[b] >= wide_tn);
        if (pair_same_n && pair_small_n && !pair_use_wide)
        {
            force_generic_pair_member[b] = true;
            force_generic_pair_member[b + 1] = true;
            ++b;
        }
    }

    for (int b = 0; b < num_projections; ++b)
    {
        if ((N_per_proj[b] % 4) != 0)
        {
            return false;
        }

        const bool use_wide = (N_per_proj[b] >= 8 * K) && (N_per_proj[b] >= wide_tn);
    const bool use_square = (!prefer_generic_pair_path) && (!force_generic_pair_member[b]) && (!use_wide) && (N_per_proj[b] >= GEMV_INT8_VNNI_SQUARE_TILE_N);
        families[b] = use_wide ? GemvBatchKernelFamily::Wide
                               : (use_square ? GemvBatchKernelFamily::Square
                                             : GemvBatchKernelFamily::Generic);
    }

    auto launch_single_projection = [&](int proj) -> bool {
        return rocmGemv_int8_int8_fp32_vnni_blockwise_scaled(
            d_A_int8,
            d_B_ptrs[proj],
            d_C_ptrs[proj],
            d_scales_A_blockwise,
            d_scales_B_ptrs[proj],
            N_per_proj[proj],
            K,
            alpha,
            beta,
            beta != 0.0f ? d_C_ptrs[proj] : nullptr,
            d_bias_ptrs[proj],
            device_id,
            stream);
    };

    auto launch_group = [&](int start_idx, int end_idx, GemvBatchKernelFamily family) -> bool {
        if ((end_idx - start_idx) == 1)
        {
            return launch_single_projection(start_idx);
        }

        // Fast path: LDS K-reduce pair kernel for tiny K/V projections.
        // Provides ~1.75x speedup over 2× individual tiny_kv launches by
        // doubling CU occupancy and eliminating launch overhead.
        // Phase 3: Removed K>=2048 gate — pair kernel works for all K values
        // (including 0.5B models with K=896). The unified KB selector handles
        // occupancy tuning for any K.
        const bool is_tiny_kv_pair =
            (family == GemvBatchKernelFamily::Generic) &&
            ((end_idx - start_idx) == 2) &&
            (N_per_proj[start_idx] <= 512) &&
            (N_per_proj[start_idx + 1] <= 512) &&
            (beta == 0.0f) &&
            (d_bias_ptrs[start_idx] == nullptr) &&
            (d_bias_ptrs[start_idx + 1] == nullptr);

        if (is_tiny_kv_pair)
        {
            return rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_pair(
                d_A_int8, d_scales_A_blockwise,
                d_B_ptrs[start_idx], d_B_ptrs[start_idx + 1],
                d_C_ptrs[start_idx], d_C_ptrs[start_idx + 1],
                d_scales_B_ptrs[start_idx], d_scales_B_ptrs[start_idx + 1],
                N_per_proj[start_idx], N_per_proj[start_idx + 1],
                K, alpha,
                device_id, stream);
        }

        const bool force_pair = g_int8_vnni_force_pair_override.load(std::memory_order_relaxed) != 0;
        const bool use_split_single_pair_path =
            !force_pair &&
            (family == GemvBatchKernelFamily::Generic) &&
            ((end_idx - start_idx) == 2) &&
            (N_per_proj[start_idx + 1] == N_per_proj[start_idx]) &&
            (N_per_proj[start_idx] <= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N) &&
            ((N_per_proj[start_idx] >= 2 * GEMV_INT8_VNNI_SQUARE_TILE_N) ||
             ((N_per_proj[start_idx] <= GEMV_INT8_VNNI_SQUARE_TILE_N) && (K >= 2048)));

        if (use_split_single_pair_path)
        {
            return launch_single_projection(start_idx) &&
                   launch_single_projection(start_idx + 1);
        }

        GemvBatchArgs args = {};
        args.num_batches = end_idx - start_idx;
        args.total_grid_n = 0;
        args.total_N = 0;

        const int tile_n = family == GemvBatchKernelFamily::Wide
            ? wide_tn
            : (family == GemvBatchKernelFamily::Square ? GEMV_INT8_VNNI_SQUARE_TILE_N
                                                       : GEMV_INT8_VNNI_GRID_KPAR_TILE_N);

        for (int src = start_idx; src < end_idx; ++src)
        {
            const int dst = src - start_idx;
            args.d_B[dst] = d_B_ptrs[src];
            args.d_C[dst] = d_C_ptrs[src];
            args.d_scales_B[dst] = d_scales_B_ptrs[src];
            args.d_bias[dst] = d_bias_ptrs[src];
            args.N[dst] = N_per_proj[src];
            args.grid_n_offset[dst] = args.total_grid_n;
            args.n_offset[dst] = args.total_N;

            args.total_grid_n += (N_per_proj[src] + tile_n - 1) / tile_n;
            args.total_N += N_per_proj[src];
        }

        if (family == GemvBatchKernelFamily::Wide)
        {
            dim3 block(wide_tn / wide_cpt, 1, 1);
            if (wide_tn == 128)
            {
                if (wide_cpt == 2)
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 2, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 2, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
                else
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 4, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<128, 4, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
            }
            else if (wide_tn == 512)
            {
                if (wide_cpt == 2)
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 2, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 2, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
                else
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 4, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<512, 4, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
            }
            else
            {
                if (wide_cpt == 2)
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 2, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 2, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
                else
                {
                    if (wide_vec_load)
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 4, true>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                    else
                    {
                        hipLaunchKernelGGL(HIP_KERNEL_NAME(gemv_int8_int8_wide_batched_vnni_blockwise_scaled_kernel_t<256, 4, false>), dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
                    }
                }
            }
        }
        else if (family == GemvBatchKernelFamily::Square)
        {
            dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
            hipLaunchKernelGGL(
                gemv_int8_int8_square_batched_vnni_blockwise_scaled_kernel,
                dim3(args.total_grid_n), block, 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_scales_A_blockwise, args, K, alpha, beta);
        }
        else
        {
            constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
            constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
            constexpr int TN_WIDE = GEMV_INT8_VNNI_WIDE2_TILE_N;
            constexpr int NUM_CUS = 60;
            constexpr int TARGET_WAVES_PER_CU = 8;
            constexpr int MIN_ACT_BLOCKS_PER_WAVE = 2;
            const int int8_tn_override = g_int8_vnni_tn_override.load(std::memory_order_relaxed);
            const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;

            int tn = TN;
            if (int8_tn_override == TN_WIDE)
            {
                tn = TN_WIDE;
            }
            else if (int8_tn_override == TN)
            {
                tn = TN;
            }

            const int grid_n = (args.total_N + tn - 1) / tn;
            int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;
            int kb_raw = max(1, (target_blocks + args.total_grid_n - 1) / args.total_grid_n);
            int kb_max = max(1, act_blocks / MIN_ACT_BLOCKS_PER_WAVE);
            kb_raw = min(kb_raw, kb_max);

            int kb = kb_raw;
            if (act_blocks % kb_raw != 0)
            {
                for (int d = 1; d < kb_raw; ++d)
                {
                    if (kb_raw - d >= 1 && act_blocks % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
                    if (kb_raw + d <= kb_max && act_blocks % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
                }
            }

            if (int8_kb_override > 0)
            {
                kb = min(int8_kb_override, act_blocks);
            }

            // Small generic batched groups (for example decode KV pairs with
            // 2x512 outputs) do not benefit from the aggressive occupancy-
            // driven split count used by the single-projection path. The
            // direct batched kernel is consistently best around KB<=4 there,
            // while larger split counts just add atomic/update overhead.
            if (args.total_grid_n <= 8 && int8_kb_override <= 0)
            {
                kb = min(kb, 4);
            }

            const bool use_pair_fast_path = (args.num_batches == 2);

            if (use_pair_fast_path)
            {
                if (tn == TN_WIDE)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_init_output_pair_kernel_t<TN_WIDE, CPT>),
                        dim3(args.total_grid_n), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                        args.d_C[0], args.d_C[1], args.d_bias[0], args.d_bias[1],
                        args.N[0], args.N[1], args.grid_n_offset[1], beta);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_init_output_pair_kernel_t<TN, CPT>),
                        dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                        args.d_C[0], args.d_C[1], args.d_bias[0], args.d_bias[1],
                        args.N[0], args.N[1], args.grid_n_offset[1], beta);
                }

                if (tn == TN_WIDE)
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_pair_vnni_blockwise_scaled_kernel_t<TN_WIDE, CPT>),
                        dim3(args.total_grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                        d_A_int8, d_scales_A_blockwise,
                        args.d_B[0], args.d_B[1],
                        args.d_C[0], args.d_C[1],
                        args.d_scales_B[0], args.d_scales_B[1],
                        args.N[0], args.N[1], args.grid_n_offset[1],
                        K, kb, alpha);
                }
                else
                {
                    hipLaunchKernelGGL(
                        HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_pair_vnni_blockwise_scaled_kernel_t<TN, CPT>),
                        dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                        d_A_int8, d_scales_A_blockwise,
                        args.d_B[0], args.d_B[1],
                        args.d_C[0], args.d_C[1],
                        args.d_scales_B[0], args.d_scales_B[1],
                        args.N[0], args.N[1], args.grid_n_offset[1],
                        K, kb, alpha);
                }
            }
            else if (tn == TN_WIDE)
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_batched_kernel_t<TN_WIDE, CPT>),
                    dim3(args.total_grid_n), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                    args, beta);

                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_batched_vnni_blockwise_scaled_kernel_t<TN_WIDE, CPT>),
                    dim3(args.total_grid_n, kb), dim3(TN_WIDE / CPT), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_scales_A_blockwise, args, K, kb, alpha);
            }
            else
            {
                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_init_output_batched_kernel_t<TN, CPT>),
                    dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    args, beta);

                hipLaunchKernelGGL(
                    HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_batched_vnni_blockwise_scaled_kernel_t<TN, CPT>),
                    dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                    d_A_int8, d_scales_A_blockwise, args, K, kb, alpha);
            }
        }

        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGemv_int8_int8_fp32_vnni_blockwise_scaled_batched] Kernel launch failed: %s (K=%d num_proj=%d subgroup=%d)\n",
                    hipGetErrorString(err), K, num_projections, args.num_batches);
            return false;
        }

        return true;
    };

    int group_start = 0;
    while (group_start < num_projections)
    {
        const GemvBatchKernelFamily family = families[group_start];
        int group_end = group_start + 1;
        while (group_end < num_projections && families[group_end] == family)
        {
            ++group_end;
        }

        if (!launch_group(group_start, group_end, family))
        {
            return false;
        }

        group_start = group_end;
    }

    return true;
}


// =============================================================================
// Fused FP32→FP32 GEMV: Quantize + GEMV + Scale in one kernel
// =============================================================================
//
// Replaces the 3-kernel decode pipeline:
//   OLD: quantizeActivations → rocmGemv_int8_int8_int32_vnni → applyScaling
//   NEW: rocmGemv_fused_fp32_int8_vnni (single launch, directly FP32→FP32)
//
// Eliminates:
//   - d_A_int8 buffer (activation quantization done in shared memory)
//   - d_scales_A buffer (scale folded into per-tile FP32 conversion)
//   - d_CK_int32 buffer (output is FP32 directly)
//   - 2 kernel launches per projection (quantize + scale)
//
bool rocmGemv_fused_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output (will be zeroed)
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    if ((N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] N must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // KB heuristic for FUSED kernel: very different from INT8 grid_kpar!
    //
    // The fused kernel uses FP32 atomicAdd for cross-block K-accumulation.
    // On gfx906, FP32 atomicAdd is emulated via CAS loop (not hardware native),
    // so ANY K-splitting causes massive contention overhead.
    //
    // Force kb=1: each block processes the entire K dimension sequentially
    // via QUANT_TILE_K=256 tiles in a loop, then does a single plain store.
    // No atomicAdd, no hipMemsetAsync needed.
    const int kb = 1;

    // Zero output only if using atomicAdd accumulation across K-blocks
    if (kb > 1)
        (void)hipMemsetAsync(d_C_fp32, 0, N * sizeof(float), static_cast<hipStream_t>(stream));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
        dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, d_bias, N, K, kb);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    // Apply bias separately if provided and using K-splitting.
    // With kb==1, bias could be fused but we keep the separate path for simplicity.
    if (d_bias)
    {
        // Reuse existing biasAdd from ROCmQuantisedGemmKernel_CK.hip
        extern bool rocmQuantGemm_biasAdd(
            float *d_output, const float *d_bias,
            int M, int N, int rocm_device_id, void *stream);

        if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
        {
            fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] biasAdd failed (N=%d)\n", N);
            return false;
        }
    }

    return true;
}


// =============================================================================
// Fused Scatter FP32→FP32 GEMV: Quantize + GEMV → partials, then reduce+scale
// =============================================================================
//
// 2-kernel pipeline replacing the 3-kernel pipeline:
//   OLD: quantizeActivations → rocmGemv_int8_int8_int32_vnni → applyScaling
//   NEW: fused_scatter (quant+GEMV→partials) → reduce+scale
//
// Eliminates:
//   - Separate quant kernel launch (~10 μs)
//   - hipMemsetAsync (~3 μs)
//   - All atomicAdd contention
//   - One kernel launch overhead (~5-8 μs)
//
// When KB=1 (enough N-blocks for full occupancy), falls back to the
// single-kernel fused path for maximum launch elimination.
//
bool rocmGemv_fused_scatter_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    float* d_partial_buf,            // [KB_MAX × N] pre-allocated partial buffer
    int* d_counter,                  // [ceil(N / TN)] pre-allocated self-reduce counters
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,       // [N] used when beta != 0
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    if ((N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] N must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // KB heuristic: same occupancy target as grid_kpar (~8 waves/CU),
    // but min KB=1 since scatter has no atomicAdd penalty for KB=1.
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_KGROUPS_PER_WAVE = 16;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
    int kb_raw = max(1, (target_blocks + grid_n - 1) / grid_n);

    int kb_max = max(1, k_groups / MIN_KGROUPS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    // Round to nearest factor of k_groups (avoid uneven work distribution)
    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb <= 1)
    {
        // Single-kernel path: enough N-blocks for occupancy with KB=1.
        // Use the existing fused kernel directly (no partial buffer needed).
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, 1), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, nullptr, N, K, 1);

        // Apply bias + alpha/beta if needed (the single-kernel path doesn't handle these)
        if (d_bias || beta != 0.0f)
        {
            // The fused kernel wrote: d_C_fp32[n] = sum * scale_B[n]
            // We need: result = alpha * sum * scale_B + beta * C_existing + bias
            // Since alpha and beta are not applied by the fused kernel for kb=1 path,
            // handle them via a quick fixup if non-trivial
            if (alpha != 1.0f || beta != 0.0f)
            {
                fprintf(stderr, "[rocmGemv_fused_scatter] alpha/beta not yet supported in kb=1 path\n");
                return false;
            }
            if (d_bias)
            {
                extern bool rocmQuantGemm_biasAdd(
                    float *d_output, const float *d_bias,
                    int M, int N, int rocm_device_id, void *stream);
                if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
                    return false;
            }
        }
    }
    else
    {
        // Hybrid dispatch: use self-reducing scatter for small N (few n-blocks),
        // 2-kernel scatter+reduce for large N.
        //
        // Rationale: For small grid_n (< 16), the reduce kernel has too few
        // blocks to fill 60 CUs → catastrophically slow (~27μs for 4 blocks).
        // Self-reduce eliminates the reduce kernel using atomicAdd counters.
        // For large grid_n, the reduce kernel is fast (~4μs) and the atomic
        // overhead in self-reduce would cause a slight regression.
        constexpr int SELFREDUCE_GRID_N_THRESHOLD = 16;

        if (grid_n < SELFREDUCE_GRID_N_THRESHOLD)
        {
            if (!d_counter)
            {
                fprintf(stderr, "[rocmGemv_fused_scatter] self-reduce counter workspace is required (N=%d K=%d)\n",
                        N, K);
                return false;
            }

            int k_per_block = (k_groups + kb - 1) / kb;
            int valid_kb = (k_groups + k_per_block - 1) / k_per_block;

            (void)hipMemsetAsync(d_counter, 0, grid_n * sizeof(int),
                           static_cast<hipStream_t>(stream));

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_selfreduce_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, d_C_fp32, d_scales_B, d_bias,
                d_counter, N, K, kb, valid_kb, alpha, beta, d_C_existing);
        }
        else
        {
            // Two-kernel path: fused scatter → reduce+scale
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, N, K, kb);

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_reduce_scale_kernel_t<TN, CPT>),
                dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_partial_buf, d_C_fp32, d_scales_B, d_bias, N, kb,
                alpha, beta, d_C_existing);
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    return true;
}


// =============================================================================
// Self-Reducing Scatter: single-kernel GEMV with last-to-arrive reduction
// =============================================================================
//
// Fuses scatter + reduce into a single kernel using atomicAdd counters.
// The last k-block to complete for each n-block performs the reduction
// inline, eliminating the reduce kernel launch entirely.
//
// Requires:
//   - d_counter: [grid_n] int array, zeroed before each launch
//   - d_partial_buf: [KB_MAX × N] float buffer for partials
//
bool rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    float* d_partial_buf,            // [KB_MAX × N] pre-allocated partial buffer
    int* d_counter,                  // [grid_n] atomic counters (zeroed before call)
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,       // [N] used when beta != 0
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0 || (N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_selfreduce] N and K must be multiples of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // Same KB heuristic as scatter dispatch
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_KGROUPS_PER_WAVE = 16;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
    int kb_raw = max(1, (target_blocks + grid_n - 1) / grid_n);

    int kb_max = max(1, k_groups / MIN_KGROUPS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    // Round to nearest factor of k_groups
    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb <= 1)
    {
        // Single-kernel path: no k-splitting needed, use existing fused kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, 1), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, nullptr, N, K, 1);

        if (alpha != 1.0f || beta != 0.0f)
        {
            fprintf(stderr, "[rocmGemv_selfreduce] alpha/beta not supported in kb=1 path\n");
            return false;
        }
        if (d_bias)
        {
            extern bool rocmQuantGemm_biasAdd(
                float *d_output, const float *d_bias,
                int M, int N, int rocm_device_id, void *stream);
            if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
                return false;
        }
    }
    else
    {
        // Compute valid_kb: the actual number of k-blocks that will participate.
        // With uniform k-splitting, all kb blocks do work when k_groups is
        // exactly divisible. The heuristic above ensures this is usually the case,
        // but guard against edge cases.
        int k_per_block = (k_groups + kb - 1) / kb;
        int valid_kb = (k_groups + k_per_block - 1) / k_per_block;

        // Hybrid dispatch: use self-reducing scatter only when grid_n is small
        // (< 16 n-blocks, i.e. N < 2048). For small grid_n, the reduce kernel
        // has too few blocks to fill 60 CUs and becomes catastrophically slow
        // (~27μs for 4 blocks). For large grid_n, the reduce kernel is fast
        // (~4μs) and avoiding atomicAdd/threadfence overhead is better.
        constexpr int SELFREDUCE_GRID_N_THRESHOLD = 16;

        if (grid_n < SELFREDUCE_GRID_N_THRESHOLD && d_counter)
        {
            // Self-reducing path: single kernel, last-to-arrive reduction
            (void)hipMemsetAsync(d_counter, 0, grid_n * sizeof(int),
                           static_cast<hipStream_t>(stream));

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_selfreduce_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, d_C_fp32, d_scales_B, d_bias,
                d_counter, N, K, kb, valid_kb, alpha, beta, d_C_existing);
        }
        else
        {
            // Two-kernel path: scatter + reduce (better for large grid_n)
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, N, K, kb);

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_reduce_scale_kernel_t<TN, CPT>),
                dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_partial_buf, d_C_fp32, d_scales_B, d_bias, N, kb,
                alpha, beta, d_C_existing);
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_selfreduce] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    return true;
}


bool rocmGemv_int8_scatter_vnni_blockwise(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scales_A_blockwise,
    const float* d_scales_B,
    const float* d_bias,
    float* d_partial_buf,
    int* d_counter,
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0 || (N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] N and K must be multiples of 4 (N=%d K=%d)\n", N, K);
        return false;
    }
    if ((K % GEMV_INT8_VNNI_ACT_BLOCK_K) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] K must be a multiple of %d for blockwise scales (K=%d)\n",
                GEMV_INT8_VNNI_ACT_BLOCK_K, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    const int grid_n = (N + TN - 1) / TN;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int kb = select_blockwise_split_count(grid_n, act_blocks, int8_kb_override);

    if (kb == 1)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_int8_direct_vnni_kernel_t<TN, CPT, true>),
            dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise,
            d_scales_B, d_bias, N, K,
            alpha, beta, d_C_existing);
    }
    else
    {
        constexpr int SELFREDUCE_GRID_N_THRESHOLD = 16;

        if (grid_n < SELFREDUCE_GRID_N_THRESHOLD)
        {
            if (!d_counter)
            {
                fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] self-reduce counter workspace is required (N=%d K=%d)\n",
                        N, K);
                return false;
            }

            int act_blocks_per_block = (act_blocks + kb - 1) / kb;
            int valid_kb = (act_blocks + act_blocks_per_block - 1) / act_blocks_per_block;

            (void)hipMemsetAsync(d_counter, 0, grid_n * sizeof(int),
                           static_cast<hipStream_t>(stream));

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_scatter_selfreduce_vnni_kernel_t<TN, CPT, true>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_partial_buf, d_C_fp32, d_scales_A_blockwise,
                d_scales_B, d_bias, d_counter, N, K, kb, valid_kb,
                alpha, beta, d_C_existing);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_scatter_vnni_kernel_t<TN, CPT, true>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_partial_buf, d_scales_A_blockwise, N, K, kb);

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_reduce_scale_kernel_t<TN, CPT>),
                dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_partial_buf, d_C_fp32, d_scales_B, d_bias, N, kb,
                alpha, beta, d_C_existing);
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    return true;
}


// =============================================================================
// Batched INT8 scatter GEMV dispatch: multiple projections in 2 kernel launches
// =============================================================================
//
// Launches a single batched scatter kernel + a single batched reduce kernel
// for up to MAX_GEMV_BATCH projections that share the same K dimension and
// pre-quantized INT8 activations.
//
// Grid for scatter: dim3(total_grid_n, KB)
// Grid for reduce:  dim3(total_grid_n)
//
// All projections share a single partial buffer with concatenated N layout:
//   partial[kb * total_N + n_offset[batch] + local_n]
//
// Advantages:
//   - QKV: 6 launches → 2 (3 scatter + 3 reduce → 1 scatter + 1 reduce)
//   - Gate/Up: 4 launches → 2
//   - Better reduce occupancy (QKV reduce: 36 blocks vs K/V alone: 4 blocks)
//
bool rocmGemv_int8_scatter_batched_vnni(
    const int8_t* d_A_int8,          // [K] pre-quantized INT8 activations
    const float* d_scale_A,          // [1] activation scale (device pointer)
    float* d_partial_buf,            // [KB_MAX × max_N] pre-allocated partial buffer
    int num_projections,             // Number of projections to batch (1..MAX_GEMV_BATCH)
    const int8_t* const* d_B_ptrs,   // [num_proj] VNNI weight pointers
    float* const* d_C_ptrs,          // [num_proj] FP32 output pointers
    const float* const* d_scales_B_ptrs, // [num_proj] weight scale pointers
    const float* const* d_bias_ptrs, // [num_proj] bias pointers (nullable entries OK)
    const int* N_per_proj,           // [num_proj] output dimension per projection
    int K,
    float alpha, float beta,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (num_projections < 1 || num_projections > MAX_GEMV_BATCH)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched] Invalid num_projections=%d (max=%d)\n",
                num_projections, MAX_GEMV_BATCH);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;

    // Build batch args
    GemvBatchArgs args = {};
    args.num_batches = num_projections;
    args.total_grid_n = 0;
    args.total_N = 0;

    for (int b = 0; b < num_projections; ++b)
    {
        if ((N_per_proj[b] % 4) != 0)
        {
            fprintf(stderr, "[rocmGemv_int8_scatter_batched] N[%d]=%d must be multiple of 4\n",
                    b, N_per_proj[b]);
            return false;
        }

        args.d_B[b] = d_B_ptrs[b];
        args.d_C[b] = d_C_ptrs[b];
        args.d_scales_B[b] = d_scales_B_ptrs[b];
        args.d_bias[b] = d_bias_ptrs[b];
        args.N[b] = N_per_proj[b];
        args.grid_n_offset[b] = args.total_grid_n;
        args.n_offset[b] = args.total_N;

        int grid_n_b = (N_per_proj[b] + TN - 1) / TN;
        args.total_grid_n += grid_n_b;
        args.total_N += N_per_proj[b];
    }

    // KB heuristic: same as individual scatter, but using total_grid_n
    const int k_groups = K / 4;
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_KGROUPS_PER_WAVE = 16;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
    int kb_raw = max(1, (target_blocks + args.total_grid_n - 1) / args.total_grid_n);

    int kb_max = max(1, k_groups / MIN_KGROUPS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb < 1) kb = 1;

    // Launch batched scatter: dim3(total_grid_n, kb)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_int8_scatter_batched_vnni_kernel_t<TN, CPT, false>),
        dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_int8, d_scale_A, d_partial_buf, args, K, kb);

    // Launch batched reduce: dim3(total_grid_n)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_reduce_scale_batched_kernel_t<TN, CPT>),
        dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_partial_buf, args, kb, alpha, beta);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched] Kernel launch failed: %s (K=%d kb=%d num_proj=%d)\n",
                hipGetErrorString(err), K, kb, num_projections);
        return false;
    }

    return true;
}


bool rocmGemv_int8_scatter_batched_vnni_blockwise(
    const int8_t* d_A_int8,
    const float* d_scales_A_blockwise,
    float* d_partial_buf,
    int num_projections,
    const int8_t* const* d_B_ptrs,
    float* const* d_C_ptrs,
    const float* const* d_scales_B_ptrs,
    const float* const* d_bias_ptrs,
    const int* N_per_proj,
    int K,
    float alpha, float beta,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (num_projections < 1 || num_projections > MAX_GEMV_BATCH)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] Invalid num_projections=%d (max=%d)\n",
                num_projections, MAX_GEMV_BATCH);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }
    if ((K % GEMV_INT8_VNNI_ACT_BLOCK_K) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] K must be a multiple of %d for blockwise scales (K=%d)\n",
                GEMV_INT8_VNNI_ACT_BLOCK_K, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
    constexpr int NUM_CUS = 60;
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    GemvBatchArgs args = {};
    args.num_batches = num_projections;
    args.total_grid_n = 0;
    args.total_N = 0;

    for (int b = 0; b < num_projections; ++b)
    {
        if ((N_per_proj[b] % 4) != 0)
        {
            fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] N[%d]=%d must be multiple of 4\n",
                    b, N_per_proj[b]);
            return false;
        }

        args.d_B[b] = d_B_ptrs[b];
        args.d_C[b] = d_C_ptrs[b];
        args.d_scales_B[b] = d_scales_B_ptrs[b];
        args.d_bias[b] = d_bias_ptrs[b];
        args.N[b] = N_per_proj[b];
        args.grid_n_offset[b] = args.total_grid_n;
        args.n_offset[b] = args.total_N;

        int grid_n_b = (N_per_proj[b] + TN - 1) / TN;
        args.total_grid_n += grid_n_b;
        args.total_N += N_per_proj[b];
    }

    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_ACT_BLOCKS_PER_WAVE = 2;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;
    int kb_raw = max(1, (target_blocks + args.total_grid_n - 1) / args.total_grid_n);

    int kb_max = max(1, act_blocks / MIN_ACT_BLOCKS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (act_blocks % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && act_blocks % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && act_blocks % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb < 1) kb = 1;

    if (int8_kb_override > 0)
    {
        kb = min(int8_kb_override, act_blocks);
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_int8_scatter_batched_vnni_kernel_t<TN, CPT, true>),
        dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_int8, d_scales_A_blockwise, d_partial_buf, args, K, kb);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_reduce_scale_batched_kernel_t<TN, CPT>),
        dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_partial_buf, args, kb, alpha, beta);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] Kernel launch failed: %s (K=%d kb=%d num_proj=%d)\n",
                hipGetErrorString(err), K, kb, num_projections);
        return false;
    }

    return true;
}


// =============================================================================
//
// Repacks weights from flat VNNI layout [K/4][N][4] to row-major [N×K]
// (equivalently [K×N] column-major for CK).
//
// VNNI layout:  vnni[(kg*N + n)*4 + i]  where kg = k/4, i = k%4
// Row-major:    rowmajor[n*K + k]       where k = kg*4 + i
//
// Mapping:  rowmajor[n*K + kg*4 + i] = vnni[(kg*N + n)*4 + i]
//
// Treat each 4-byte pack as one int32 element in matrix A[kg, n] and transpose
// to B[n, kg] using a tiled shared-memory transpose to keep both reads and
// writes coalesced.
//
// Cost: ~65-200μs for 18944×3584 (Gate/Up shape). Amortized over prefill.
//
__global__ void
repack_vnni_to_rowmajor_kernel(
    const int8_t* __restrict__ d_B_vnni,     // [K/4][N][4] input
    int8_t* __restrict__ d_B_rowmajor,       // [N×K] output
    int N, int K)
{
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int k_groups = K / 4; // number of int32 packs per output row

    const int x_in = blockIdx.x * TILE_DIM + threadIdx.x; // n index in source A[kg, n]
    const int y_in = blockIdx.y * TILE_DIM + threadIdx.y; // kg index in source A[kg, n]

    const int32_t* __restrict__ src32 = reinterpret_cast<const int32_t*>(d_B_vnni);
    int32_t* __restrict__ dst32 = reinterpret_cast<int32_t*>(d_B_rowmajor);

    __shared__ int32_t tile[TILE_DIM][TILE_DIM + 1];

    // Load tile from source (A = [k_groups, N], row-major in n)
    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_in + j;
        if (x_in < N && y < k_groups)
        {
            tile[threadIdx.y + j][threadIdx.x] = src32[static_cast<int64_t>(y) * N + x_in];
        }
    }

    __syncthreads();

    // Store transposed tile to destination (B = [N, k_groups], row-major in kg)
    const int x_out = blockIdx.y * TILE_DIM + threadIdx.x; // kg
    const int y_out = blockIdx.x * TILE_DIM + threadIdx.y; // n

    #pragma unroll
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        const int y = y_out + j;
        if (x_out < k_groups && y < N)
        {
            dst32[static_cast<int64_t>(y) * k_groups + x_out] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}


bool rocmGemv_repackVNNI_to_rowmajor(
    const int8_t* d_B_vnni,
    int8_t* d_B_rowmajor,
    int N, int K,
    int device_id, void* stream)
{
    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int k_groups = K / 4;
    constexpr int TILE_DIM = 32;
    constexpr int BLOCK_ROWS = 8;

    const int grid_x = (N + TILE_DIM - 1) / TILE_DIM;
    const int grid_y = (k_groups + TILE_DIM - 1) / TILE_DIM;

    hipLaunchKernelGGL(
        repack_vnni_to_rowmajor_kernel,
        dim3(grid_x, grid_y), dim3(TILE_DIM, BLOCK_ROWS), 0, static_cast<hipStream_t>(stream),
        d_B_vnni, d_B_rowmajor, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

/**
 * @brief Entrypoint for INT8 VNNI prefill GEMM.
 *
 * This function dispatches to tiled INT8-x-INT8 GEMM kernels for M>1 prefill.
 * Returning false instructs the caller to keep using the CK fallback path.
 *
 * Junior-dev note:
 * - The decode path (M=1) already has specialized GEMV kernels.
 * - Prefill path (M>1) uses tiled INT8 VNNI GEMM kernels dispatched here.
 */
enum PrefillInt8TileVariant : int
{
    PREFILL_TILE_16x16 = 0,
    PREFILL_TILE_32x8 = 1,
    PREFILL_TILE_8x32 = 2,
    PREFILL_TILE_8x8 = 3,
};

static const char *prefillTileVariantName(int variant)
{
    switch (variant)
    {
    case PREFILL_TILE_16x16:
        return "16x16";
    case PREFILL_TILE_32x8:
        return "32x8";
    case PREFILL_TILE_8x32:
        return "8x32";
    case PREFILL_TILE_8x8:
        return "8x8";
    default:
        return "unknown";
    }
}

static int select_prefill_int8_tile_variant_auto(int M, int N, int K, bool grid_kpar)
{
    (void)K;
    if (M <= 12 && N <= 512)
    {
        return PREFILL_TILE_8x8;
    }

    if (N >= 4 * M && N >= 1024)
    {
        return PREFILL_TILE_32x8;
    }

    if (M >= 4 * N && M >= 128)
    {
        return PREFILL_TILE_8x32;
    }

    if (grid_kpar && N >= 2 * M && N >= 2048)
    {
        return PREFILL_TILE_32x8;
    }

    return PREFILL_TILE_16x16;
}

static int resolve_prefill_int8_tile_variant(int forced_variant, int M, int N, int K, bool grid_kpar)
{
    if (forced_variant >= PREFILL_TILE_16x16 && forced_variant <= PREFILL_TILE_8x8)
    {
        return forced_variant;
    }
    return select_prefill_int8_tile_variant_auto(M, N, K, grid_kpar);
}

bool rocmGemm_int8_int8_int32_vnni_prefill(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, false);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
        }
    }

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                hipGetErrorString(err), M, N, K);
        return false;
    }

    return true;
}

/**
 * @brief INT8 VNNI prefill grid-kpar split-K variant.
 *
 * This variant uses grid.z to split K reduction into slices with INT32
 * atomic accumulation. It is opt-in from higher-level dispatch.
 */
bool rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int M, int N, int K,
    int split_k_slices,
    int tile_variant,
    int cpt,
    int device_id, void* stream)
{
    if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
        return false;
    }

    const int total_k_groups = K / 4;
    const int slices = max(1, min(split_k_slices, total_k_groups));
    const int k_groups_per_slice = (total_k_groups + slices - 1) / slices;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
    hipError_t memset_err = hipMemsetAsync(d_C_int32, 0, c_bytes, static_cast<hipStream_t>(stream));
    if (memset_err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] hipMemsetAsync failed: %s\n", hipGetErrorString(memset_err));
        return false;
    }

    const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, true);
    const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

    if (variant == PREFILL_TILE_32x8)
    {
        constexpr int BX = 32;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x32)
    {
        constexpr int BX = 8;
        constexpr int BY = 32;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else if (variant == PREFILL_TILE_8x8)
    {
        constexpr int BX = 8;
        constexpr int BY = 8;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }
    else
    {
        constexpr int BX = 16;
        constexpr int BY = 16;
        const dim3 block(BX, BY, 1);
        if (cpt_v == 4)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4>),
                               dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else if (cpt_v == 2)
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2>),
                               dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
        else
        {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(gemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1>),
                               dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), static_cast<unsigned>(slices)),
                               block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice);
        }
    }


    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemm_int8_int8_int32_vnni_prefill_grid_kpar] Kernel launch failed: %s (M=%d N=%d K=%d slices=%d)\n",
                hipGetErrorString(err), M, N, K, slices);
        return false;
    }

    return true;
}

} // extern "C"
