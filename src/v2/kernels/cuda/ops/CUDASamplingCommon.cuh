/**
 * @file CUDASamplingCommon.cuh
 * @brief Shared CUDA sampling constants, argmax/top-k device helpers, and speculative threshold helper.
 */
#pragma once
/**
 * @file CUDASamplingKernels.cu
 * @brief CUDA GPU-side sampling kernels for argmax and top-k
 *
 * Provides GPU-side argmax and top-k selection over FP32 logits.
 * Mirrors the ROCm implementations in ROCmArgmaxKernels.hip and
 * ROCmSamplingKernels.hip for cross-backend parity.
 * Ordinary and speculative generation share the SamplingMath controller ABI:
 * immutable admission selects the algorithm, and explicit-stream publication
 * writes persistent response rows without host sampling or token transfers.
 *
 * Kernels:
 *   - Argmax: Two-pass multi-block reduction (pass 1 spreads the vocab across
 *     many blocks/SMs producing one partial per block; pass 2 reduces the
 *     partials in a single small block). Falls back to a single-block reduction
 *     when no partial scratch is supplied.
 *   - Top-K: deterministic value/index ordering. Qwen-sized batched rows use
 *     a two-stage cooperative register selector; larger geometries retain a
 *     total generic partial-list implementation.
 */

#include <cuda_runtime.h>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include "../../common/SamplingMath.h"
#include "../../common/GenerationLogicalState.h"
#include "../../common/SamplingWideTopKDevice.inl"
#include "../../../execution/mtp/MTPVerifierOutcomeGraph.h"
#include "../../../execution/moe/DeviceMoERebalanceABI.h"

// Maximum k supported by the top-k kernel
constexpr int TOPK_MAX_K = 256;
constexpr int TOPK_THREADS = 32;
// Qwen chat defaults use top_k=40. A 40-entry specialization keeps register
// pressure aligned with the actual compact table size used by stochastic MTP.
constexpr int TOPK_MEDIUM_K_CAP = 40;
constexpr int TOPK_SMALL_K_CAP = 64;
constexpr int TOPK_SMALL_K_PARTIAL_BLOCKS = 128;
// Qwen-style top_k=40/64 distributions expose enough independent partials to
// cover one full GA102 while keeping each segment within the cooperative
// selector's fixed register capacity.
constexpr int TOPK_SMALL_K_WIDE_PARTIAL_BLOCKS = 64;
constexpr int TOPK_SMALL_K_THREADS = 64;
constexpr int TOPK_BATCHED_COOPERATIVE_THREADS = 256;
constexpr int TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD = 16;
constexpr int TOPK_BATCHED_COOPERATIVE_CAPACITY =
    TOPK_BATCHED_COOPERATIVE_THREADS *
    TOPK_BATCHED_COOPERATIVE_ITEMS_PER_THREAD;
constexpr int TOPK_BATCHED_COOPERATIVE_WARPS =
    TOPK_BATCHED_COOPERATIVE_THREADS / 32;
static_assert(TOPK_MAX_K == llaminar2::sampling_math::kMaxTopK,
              "CUDA sampling TOPK_MAX_K must match shared sampling math");

/**
 * @brief Opt in to the generic Top-K kernel's full, fixed shared-memory envelope.
 *
 * CUDA's default per-block allowance is 48 KiB; 32 sorted lists at K=256
 * require 64 KiB. This is launch preparation, like the attention kernel's
 * shared-memory admission, not captured execution. Every caller selects the
 * same maximum attribute, so concurrent captures never shrink one another's
 * envelope. No extra global memory, new kernel or arithmetic change is needed.
 * Reapplying is intentional: runtime reclamation may recreate the CUDA context.
 * @param kernel Exact specialization whose immutable resource limit is admitted.
 * @param requested_bytes Dynamic shared bytes embedded in the upcoming launch.
 * @return Whether the supported envelope was accepted, with a precise CUDA error otherwise.
 */
template <typename Kernel>
static bool prepareTopKSharedMemory(Kernel kernel, size_t requested_bytes)
{
    constexpr size_t default_bytes = 48u * 1024u;
    constexpr int maximum_bytes = TOPK_THREADS * TOPK_MAX_K * (sizeof(float) + sizeof(int));
    if (requested_bytes <= default_bytes)
        return true;
    const auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, maximum_bytes);
    if (error == cudaSuccess)
        return true;
    fprintf(stderr, "CUDA Top-K shared-memory admission (%d bytes) failed: %s\n",
        maximum_bytes, cudaGetErrorString(error));
    return false;
}

constexpr int topkSmallKPartialBlockCap(int k)
{
    return k > 32 ? TOPK_SMALL_K_WIDE_PARTIAL_BLOCKS : TOPK_SMALL_K_PARTIAL_BLOCKS;
}

/**
 * @brief Deterministic ordering for Top-K candidates.
 *
 * A parallel reduction should not let equal logits depend on thread traversal
 * order.  We therefore sort by logit descending and token id ascending.  This
 * is the same contract as the ROCm sampler and keeps seeded stochastic decode
 * reproducible when quantized logits contain ties.
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

// ── Argmax multi-block reduction tuning ─────────────────────────────────────
// Threads per block for both reduction passes. Every production geometry is a
// power-of-two multiple of one CUDA warp so the deterministic warp reduction
// below can remove repeated block-wide barriers.
constexpr int ARGMAX_REDUCE_THREADS = 256;
constexpr int ARGMAX_FINALIZE_THREADS = 256;
// Target elements processed per thread in pass 1.  The production Qwen 3.6
// vocabulary has 248,320 columns, so four elements per thread launches 243
// blocks at the 256-thread geometry.  A repeated full-matrix Perf__ sweep on an
// RTX 3090 selected this geometry over the lower-wave 8-element launch while
// preserving the serial lower-token tie break exactly.
constexpr int ARGMAX_ELEMS_PER_THREAD = 4;



__device__ __forceinline__ bool argmax_better(float candidate_value,
                                              int candidate_index,
                                              float current_value,
                                              int current_index)
{
    return candidate_value > current_value ||
           (candidate_value == current_value && candidate_index < current_index);
}

/**
 * @brief Value/index pair ordered by serial argmax semantics.
 *
 * The index tie break is part of correctness, not merely determinism: serial
 * decode retains the lowest token id among equal logits.  Carrying both words
 * through every shuffle makes the parallel grouping mathematically identical
 * to that order for all finite FP32 logits.
 */
struct CudaArgmaxPair
{
    float value;
    int index;
};

/**
 * @brief Reduce one full CUDA warp without shared memory or barriers.
 */
__device__ __forceinline__ CudaArgmaxPair cuda_argmax_warp_reduce(
    CudaArgmaxPair pair)
{
    constexpr unsigned kFullWarpMask = 0xffffffffU;
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        const CudaArgmaxPair candidate{
            __shfl_down_sync(kFullWarpMask, pair.value, offset),
            __shfl_down_sync(kFullWarpMask, pair.index, offset)};
        if (argmax_better(
                candidate.value,
                candidate.index,
                pair.value,
                pair.index))
        {
            pair = candidate;
        }
    }
    return pair;
}

/**
 * @brief Reduce one block with one barrier and one final warp reduction.
 *
 * All supported launch geometries contain complete warps.  Each warp first
 * reduces independently, lane zero publishes one pair to shared memory, and
 * warp zero performs the final reduction.  Only thread zero consumes the
 * returned pair, so no trailing block barrier is necessary.
 */
__device__ __forceinline__ CudaArgmaxPair cuda_argmax_block_reduce(
    CudaArgmaxPair pair,
    float *warp_values,
    int *warp_indices)
{
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int warp_count = static_cast<int>(blockDim.x) / warpSize;

    pair = cuda_argmax_warp_reduce(pair);
    if (lane == 0)
    {
        warp_values[warp] = pair.value;
        warp_indices[warp] = pair.index;
    }
    __syncthreads();

    if (warp == 0)
    {
        pair = lane < warp_count
                   ? CudaArgmaxPair{
                         warp_values[lane],
                         warp_indices[lane]}
                   : CudaArgmaxPair{-FLT_MAX, INT_MAX};
        pair = cuda_argmax_warp_reduce(pair);
    }
    return pair;
}


static llaminar2::sampling_math::SpeculativeBatchThresholdParameters
packCudaSpeculativeBatchThresholds(
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

/**
 * @brief Validate a launch width accepted by the warp-synchronous reducer.
 */

