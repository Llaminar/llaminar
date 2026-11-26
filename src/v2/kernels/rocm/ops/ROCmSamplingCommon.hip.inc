/**
 * @file ROCmSamplingCommon.hip.inc
 * @brief Shared ROCm sampling constants, top-k device helpers, and speculative threshold helper.
 */
#pragma once
/**
 * @file ROCmSamplingKernels.hip
 * @brief ROCm/HIP resident sampling and generation-controller kernels.
 *
 * FP32 vocabulary selection, distribution processing and speculative rejection
 * retain their outputs in persistent device buffers. Ordinary and speculative
 * response publication use the same SamplingMath controller and exact integer
 * transitions as CUDA and CPU tests. Every launch requires the caller's exact
 * non-null stream; admission supplies immutable policy, not a host shadow of
 * live budgets. Captured kernels perform no allocation or host token transfer.
 */

#include <hip/hip_runtime.h>
#include <cstddef>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../execution/moe/DeviceMoERebalanceABI.h"
#include "../../../execution/mtp/MTPVerifierOutcomeGraph.h"
#include "../../../utils/DebugEnv.h"
#include "../../common/SamplingMath.h"
#include "../../common/GenerationLogicalState.h"
#include "../../common/SamplingWideTopKDevice.inl"

// Maximum k supported by the kernel (limited by shared memory and registers)
constexpr int TOPK_MAX_K = 256;
static_assert(TOPK_MAX_K == llaminar2::sampling_math::kMaxTopK,
              "ROCm sampling TOPK_MAX_K must match shared sampling math");
// Qwen chat defaults use top_k=40. Give that very common medium-k path its
// own template cap so HIP does not reserve 64-entry local arrays per thread.
constexpr int TOPK_MEDIUM_K_CAP = 40;
constexpr int TOPK_SMALL_K_CAP = 64;
constexpr int TOPK_SMALL_K_PARTIAL_BLOCKS = 128;
// Qwen-style top_k=40/64 distributions need enough partial blocks to occupy
// MI50's 60 CUs.  The cooperative selector below needs every block-owned span
// and final partial set to fit its register-resident candidate inventory.
constexpr int TOPK_SMALL_K_WIDE_PARTIAL_BLOCKS = 64;
constexpr int TOPK_SMALL_K_THREADS = 64;
constexpr int TOPK_BATCHED_COOPERATIVE_THREADS = 256;
constexpr int TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD = 16;
constexpr int TOPK_BATCHED_COOPERATIVE_CAPACITY =
    TOPK_BATCHED_COOPERATIVE_THREADS *
    TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
constexpr int TOPK_BATCHED_COOPERATIVE_WAVES =
    TOPK_BATCHED_COOPERATIVE_THREADS / 64;

// Number of threads per block — one warp on NVIDIA, half wavefront on AMD
// Using 32 threads keeps register pressure manageable for large k
constexpr int TOPK_THREADS = 32;



constexpr int topkSmallKPartialBlockCap(int k)
{
    return k > 32 ? TOPK_SMALL_K_WIDE_PARTIAL_BLOCKS : TOPK_SMALL_K_PARTIAL_BLOCKS;
}

constexpr int topkSmallKPartialBlockCap(int k, int row_count)
{
    (void)row_count;
    return topkSmallKPartialBlockCap(k);
}

static inline int runtimeTopkSmallKPartialBlockCap(int k)
{
    const int override_cap = llaminar2::debugEnv().rocm.topk_smallk_partial_blocks;
    return override_cap > 0 ? override_cap : topkSmallKPartialBlockCap(k);
}

static inline int runtimeTopkSmallKPartialBlockCap(int k, int row_count)
{
    const int override_cap = llaminar2::debugEnv().rocm.topk_smallk_partial_blocks;
    return override_cap > 0 ? override_cap : topkSmallKPartialBlockCap(k, row_count);
}

/**
 * @brief Return the complete LDS footprint for the batched small-k pipeline.
 *
 * Every lane owns one sorted K-entry candidate list. Keeping those lists in
 * explicit LDS is important on gfx906: an automatic `float[K]` plus `int[K]`
 * pair spills hundreds of bytes per lane to device memory at top_k=40. The
 * second-stage merge additionally owns three K-entry arrays for the merged
 * values, token ids, and probability weights. Both
 * kernels receive this same conservative allocation so their layout and launch
 * contract cannot drift apart.
 */
constexpr size_t batchedTopkSmallKSharedBytes(int threads, int k)
{
    const size_t lane_candidate_bytes =
        static_cast<size_t>(threads) * static_cast<size_t>(k) *
        (sizeof(float) + sizeof(int));
    const size_t merged_candidate_and_weight_bytes =
        static_cast<size_t>(k) *
        (2 * sizeof(float) + sizeof(int));
    return lane_candidate_bytes + merged_candidate_and_weight_bytes;
}

/**
 * @brief Deterministic ordering for Top-K candidates.
 *
 * GPU reductions visit candidates in backend-specific thread/block order.  For
 * repeated deterministic decoding we still need equal logits to resolve the
 * same way every time, so ties are ordered by the smaller token id.  Invalid
 * candidates use a negative token id and always lose.
 */
__device__ __forceinline__ bool topkCandidateBetter(
    float candidate_value,
    int candidate_index,
    float current_value,
    int current_index)
{
    if (candidate_index < 0)
        return false;
    if (current_index < 0)
        return true;
    return candidate_value > current_value ||
           (candidate_value == current_value &&
            candidate_index < current_index);
}

/**
 * @brief One exact Top-K candidate carried through a wave64 reduction.
 */
struct ROCmTopKCandidate
{
    float value;
    int index;
};

/**
 * @brief Small block-shared rendezvous for cooperative Top-K selection.
 *
 * Candidate payloads stay in lane registers.  LDS carries only one winner per
 * wave, the selected block winner, and the final K entries, reducing the
 * Qwen Top-K 40 footprint from 20.5 KiB to a few hundred bytes per workgroup.
 */
template <int K_CAP>
struct ROCmBatchedTopKCooperativeStorage
{
    float wave_values[TOPK_BATCHED_COOPERATIVE_WAVES];
    int wave_indices[TOPK_BATCHED_COOPERATIVE_WAVES];
    int winner_index;
    float selected_values[K_CAP];
    int selected_indices[K_CAP];
};

/**
 * @brief Reduce one deterministic candidate across a complete gfx906 wave.
 *
 * Every edge uses the canonical total comparator.  The reduction changes no
 * floating-point arithmetic because Top-K performs only exact loads and
 * comparisons; equal logits continue to select the lower token id.
 */
__device__ __forceinline__ ROCmTopKCandidate rocmTopKReduceWaveBest(
    ROCmTopKCandidate candidate)
{
    const int lane = static_cast<int>(threadIdx.x) & 63;
    for (int offset = 32; offset > 0; offset >>= 1)
    {
        const float peer_value = __shfl_down(candidate.value, offset, 64);
        const int peer_index = __shfl_down(candidate.index, offset, 64);
        if (lane + offset < 64 &&
            topkCandidateBetter(
                peer_value,
                peer_index,
                candidate.value,
                candidate.index))
        {
            candidate.value = peer_value;
            candidate.index = peer_index;
        }
    }
    return candidate;
}

/**
 * @brief Select a sorted block-wide Top-K from register-resident candidates.
 *
 * Each of 256 lanes owns sixteen candidate slots.  For each requested rank,
 * lanes select their private maximum, four wave64 reductions publish wave
 * maxima, and wave zero chooses the block winner.  Only the owning lane then
 * invalidates that token id.  This fixed procedure replaces divergent
 * insertion sorts and large lane-private LDS lists without atomics, scratch
 * allocation, synchronization outside the workgroup, or order-dependent math.
 */
template <int K_CAP>
__device__ __forceinline__ void rocmSelectBatchedTopKCooperative(
    float (&local_values)[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD],
    int (&local_indices)[TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD],
    int k,
    ROCmBatchedTopKCooperativeStorage<K_CAP> &storage)
{
    const int lane = static_cast<int>(threadIdx.x) & 63;
    const int wave = static_cast<int>(threadIdx.x) >> 6;

    for (int rank = 0; rank < k; ++rank)
    {
        ROCmTopKCandidate local_best{-FLT_MAX, -1};
#pragma unroll
        for (int item = 0;
             item < TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
             ++item)
        {
            if (topkCandidateBetter(
                    local_values[item],
                    local_indices[item],
                    local_best.value,
                    local_best.index))
            {
                local_best.value = local_values[item];
                local_best.index = local_indices[item];
            }
        }

        const ROCmTopKCandidate wave_best =
            rocmTopKReduceWaveBest(local_best);
        if (lane == 0)
        {
            storage.wave_values[wave] = wave_best.value;
            storage.wave_indices[wave] = wave_best.index;
        }
        __syncthreads();

        if (wave == 0)
        {
            ROCmTopKCandidate block_best{-FLT_MAX, -1};
            if (lane < TOPK_BATCHED_COOPERATIVE_WAVES)
            {
                block_best.value = storage.wave_values[lane];
                block_best.index = storage.wave_indices[lane];
            }
            block_best = rocmTopKReduceWaveBest(block_best);
            if (lane == 0)
            {
                storage.winner_index = block_best.index;
                storage.selected_values[rank] = block_best.value;
                storage.selected_indices[rank] = block_best.index;
            }
        }
        __syncthreads();

        const int winner_index = storage.winner_index;
#pragma unroll
        for (int item = 0;
             item < TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
             ++item)
        {
            if (local_indices[item] == winner_index)
                local_indices[item] = -1;
        }
    }
}

/**
 * @brief Merge one sorted candidate list per wave lane into a global Top-K.
 *
 * Each lane owns a descending list whose arithmetic-free insertion ordering is
 * already canonical. For every output rank, a six-step wave64 shuffle tree
 * selects the best current head with topkCandidateBetter(), then only the
 * winning lane advances its private cursor. This replaces the historical
 * lane-zero scan over all 64 heads. Although the reduction topology changes,
 * the comparator is a total order over exact FP32 values and token ids, so the
 * selected sequence retains the canonical lower-token tie break byte for byte.
 * The caller must execute one full gfx906 wave and publish its LDS lane lists
 * before entering this routine.
 *
 * @param lane_values Calling lane's sorted candidate values in LDS.
 * @param lane_indices Calling lane's matching token ids in LDS.
 * @param lane_count Number of valid candidates owned by the calling lane.
 * @param k Number of global candidates to publish.
 * @param out_values Common output array; lane zero is its sole writer.
 * @param out_indices Common token-id array; lane zero is its sole writer.
 */
__device__ __forceinline__ void mergeTopkWave64LaneLists(
    const float *lane_values,
    const int *lane_indices,
    int lane_count,
    int k,
    float *out_values,
    int *out_indices)
{
    const int lane = static_cast<int>(threadIdx.x);
    int cursor = 0;

    for (int rank = 0; rank < k; ++rank)
    {
        float winner_value =
            cursor < lane_count ? lane_values[cursor] : -FLT_MAX;
        int winner_index =
            cursor < lane_count ? lane_indices[cursor] : -1;
        int winner_lane = lane;

        for (int offset = TOPK_SMALL_K_THREADS / 2;
             offset > 0;
             offset >>= 1)
        {
            const float peer_value = __shfl_down(
                winner_value,
                offset,
                TOPK_SMALL_K_THREADS);
            const int peer_index = __shfl_down(
                winner_index,
                offset,
                TOPK_SMALL_K_THREADS);
            const int peer_lane = __shfl_down(
                winner_lane,
                offset,
                TOPK_SMALL_K_THREADS);

            if (lane + offset < TOPK_SMALL_K_THREADS &&
                topkCandidateBetter(
                    peer_value,
                    peer_index,
                    winner_value,
                    winner_index))
            {
                winner_value = peer_value;
                winner_index = peer_index;
                winner_lane = peer_lane;
            }
        }

        const int selected_lane = __shfl(
            winner_lane,
            0,
            TOPK_SMALL_K_THREADS);
        if (lane == 0)
        {
            out_values[rank] = winner_value;
            out_indices[rank] = winner_index;
        }
        if (winner_index >= 0 && lane == selected_lane)
            ++cursor;
    }
}


static llaminar2::sampling_math::SpeculativeBatchThresholdParameters
packROCmSpeculativeBatchThresholds(
    const float *accept_thresholds,
    const float *residual_thresholds,
    int row_count)
{
    llaminar2::sampling_math::SpeculativeBatchThresholdParameters packed{};
    for (int row = 0; row < row_count; ++row)
    {
        if (accept_thresholds)
            packed.accept[row] = accept_thresholds[row];
        if (residual_thresholds)
            packed.residual[row] = residual_thresholds[row];
    }
    return packed;
}


