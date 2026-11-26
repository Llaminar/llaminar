/**
 * @file ROCmGemvKernel_INT8_VNNI_common.hip.inc
 * @brief Shared INT8 VNNI constants, override declarations, split-K selectors, and inline device helpers.
 */
#pragma once
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
 * @see ROCmGemvKernel_native_quantised.hip for native-quantised 4-bit kernels
 * Native signed INT8 dot products share the exact unsaturated arithmetic
 * contract across gfx906 and gfx1100; lane ownership and reductions are unchanged.
 */

#include "kernels/rocm/common/ROCmSignedDot4.h"
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

namespace llaminar2::rocm::int8_vnni_internal {
extern std::atomic<int> g_int8_vnni_tn_override;
extern std::atomic<int> g_int8_vnni_kb_override;
extern std::atomic<int> g_int8_vnni_wk_override;
extern std::atomic<int> g_int8_vnni_unroll_override;  // 1=standard, 2=2x act_block unroll
extern std::atomic<int> g_int8_vnni_act_block_override;  // -1=default(32), 32, 64, 128
extern std::atomic<int> g_int8_vnni_wide_tn_override;
extern std::atomic<int> g_int8_vnni_wide_cpt_override;
extern std::atomic<int> g_int8_vnni_wide_vec_load_override;
extern std::atomic<int> g_int8_vnni_skip_memset_override;  // 1 = skip hipMemsetAsync (for benchmarking)
extern std::atomic<int> g_int8_vnni_force_pair_override;  // 1 = force batched pair path (skip split-to-singles)
}
using namespace llaminar2::rocm::int8_vnni_internal;

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
            acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, acc[0]);
            acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, acc[1]);
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int32_t b_packed = *(b_row + kg * b_stride + c);
                acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
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

/**
 * @brief Initialize each column's beta and bias before its split-K producers run.
 * @tparam TILE_N Columns owned by the CTA.
 * @tparam CPT Adjacent columns owned by each lane.
 * @param d_C_fp32 Final column bank, which may also supply the old values.
 * @param d_C_existing Original beta operand.
 * @param d_bias Optional bias.
 * @param N Number of live columns.
 * @param beta Original multiplier of C.
 *
 * The single-CTA and wide terminal leaves likewise admit the same C pointer
 * for input and output. Each owner reads its old column before its first
 * write. Split initialization completes before any arithmetic CTA consumes C.
 */
