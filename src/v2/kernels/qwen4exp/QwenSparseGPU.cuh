/**
 * @file QwenSparseGPU.cuh
 * @brief Symmetric CUDA/HIP sparse-indexer and native selected-KV kernels.
 *
 * All kernels consume execution-owned dynamic replay counts. Indexer scoring
 * uses fixed 32-lane groups on both warp32 and wave64 hardware. Integer rank
 * selection uses a fixed threshold search and ascending compaction; its exact
 * tie order does not depend on scheduling. Core attention only dereferences
 * selected rows and has the same reduction topology for M=1 and grouped M.
 */
#pragma once
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif
#include "QwenSparseMath.h"
#include "QwenSparseBoundedSelection.h"
#include <cmath>
#include <stdexcept>

#if defined(__HIPCC__)
namespace llaminar2::qwen4exp::rocm_sparse_detail
#else
namespace llaminar2::qwen4exp::cuda_sparse_detail
#endif
{
/** @brief Terminate corrupted dynamic state on the device that owns it. */
__device__ __forceinline__ void require(bool condition)
{
    if (!condition)
    {
#if defined(__HIPCC__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
}
/** @brief Fixed 32-lane reduction edge on CUDA warp32 and HIP wave64. */
__device__ __forceinline__ float shuffleDown(float value, int distance)
{
#if defined(__HIPCC__)
    return __shfl_down(value, distance, 32);
#else
    return __shfl_down_sync(0xffffffffu, value, distance, 32);
#endif
}
/** @brief Authenticate the causal transaction without a host readback. */
__device__ __forceinline__ QwenSparseReplay replay(const QwenSparseReplay *state, int rows, int tokens)
{
    const auto r = *state;
    require(r.prefix_tokens >= 0 && r.live_rows >= 0 && r.live_rows <= rows &&
            static_cast<long long>(r.prefix_tokens) + r.live_rows <= tokens && r.position_origin >= 0 &&
            static_cast<long long>(r.position_origin) + r.prefix_tokens + r.live_rows <= 0x7fffffff);
    return r;
}
/** @brief Freeze only original live tail payload plus required immutable metadata. */
__global__ __launch_bounds__(128,2) void snapshotPrior(QwenSparsePriorSnapshot p)
{
    const auto original=replay(p.replay,p.row_capacity,p.token_capacity);
    const int rows=original.live_rows ? original.prefix_tokens%4 : 0;
    const int d=int(threadIdx.x);
    for(int row=0;row<rows;++row)p.output->partial[row*128+d]=p.accepted_partial[row*128+d];
    if(d==0) {
        p.output->replay=original;p.output->row_capacity=p.row_capacity;
        *p.traffic={16u,static_cast<std::uint32_t>(rows*128*sizeof(float))};
        if(p.work_counters.bound()) {
            require(addNativeCounter64(p.work_counters.metric(QwenSparseWorkCounter::PriorHeader),16u));
            require(addNativeCounter64(p.work_counters.metric(QwenSparseWorkCounter::PriorPayload),std::uint64_t(rows)*128*sizeof(float)));
        }
    }
}
/** @brief Normalize and rotate one complete 128-coordinate index head. */
__device__ __forceinline__ void prepareHead(float value, const float *weight, int position,
    const QwenSparseIndexer &p, float *output, float *shared)
{
    const int d = int(threadIdx.x);
    const auto repr = p.representation;
    value = detail::roundRepresentation(value, repr);
    shared[d] = value * value;
    __syncthreads();
    for (int width = 64; width; width /= 2)
    {
        if (d < width) shared[d] += shared[d + width];
        __syncthreads();
    }
    const float inverse = 1.0f / sqrtf(shared[0] / 128.0f + p.norm_epsilon);
    const float normalized = detail::roundRepresentation((value * inverse) * weight[d], repr);
    __syncthreads();
    shared[d] = normalized;
    __syncthreads();
    float result = normalized;
    if (d < p.rotary_dim)
    {
        const int half = p.rotary_dim / 2;
        const int pair = d % half;
        float nativeCosine,nativeSine;
        detail::rotaryFactors(p.rotary_source,p.cosine,p.sine,position,pair,p.rotary_dim,p.rope_theta,nativeCosine,nativeSine,reinterpret_cast<const float *>(&p.inverse_frequencies));
        const float cosine = detail::roundRepresentation(nativeCosine, repr);
        const float sine = detail::roundRepresentation(nativeSine, repr);
        const float partner = d < half ? -shared[d + half] : shared[d - half];
        const float first = detail::roundRepresentation(normalized * cosine, repr);
        const float second = detail::roundRepresentation(partner * sine, repr);
        result = detail::roundRepresentation(first + second, repr);
    }
    require(isfinite(result));
    output[d] = result;
}
// CUDA's grid-constant argument retains the immutable frequency array in
// parameter memory; taking its pair-indexed address must not create a private
// per-thread copy. HIP naturally addresses the by-value kernarg directly.
#if defined(__CUDACC__) && !defined(__HIPCC__)
#define QSA_GRID_CONSTANT const __grid_constant__
#else
#define QSA_GRID_CONSTANT const
#endif
/** @brief Prepare all active indexer Q heads without writing inactive scratch. */
__global__ __launch_bounds__(128, 2) void prepareQueries(QSA_GRID_CONSTANT QwenSparseIndexer p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    require(!p.transaction_prior || p.transaction_prior->row_capacity==p.row_capacity);
    const int row_head = int(blockIdx.x);
    if (row_head >= r.live_rows * 4) return;
    const int position = r.position_origin + r.prefix_tokens + row_head / 4;
    require(position >= 0 && (p.rotary_source!=QwenSparseRotarySource::TablePlanes || position<p.position_capacity));
    __shared__ float shared[128];
    prepareHead(p.query[std::size_t(row_head) * 128 + threadIdx.x], p.query_norm, position,
                p, p.prepared_query + std::size_t(row_head) * 128, shared);
}
/** @brief Prepare only pools completed by this transaction, reading the prior raw tail. */
__global__ __launch_bounds__(128, 2) void preparePools(QSA_GRID_CONSTANT QwenSparseIndexer p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    require(!p.transaction_prior || p.transaction_prior->row_capacity==p.row_capacity);
    const int block = r.prefix_tokens / 4 + int(blockIdx.x);
    if (block >= (r.prefix_tokens + r.live_rows) / 4) return;
    const int position = r.position_origin + block * 4;
    require(position >= 0 && (p.rotary_source!=QwenSparseRotarySource::TablePlanes || position<p.position_capacity) && block < p.block_capacity);
    const int d = int(threadIdx.x);
    float pooled = 0.0f;
    for (int member = 0; member < 4; ++member)
    {
        const int token = block * 4 + member;
        const float raw = token < r.prefix_tokens
            ? p.partial_keys[std::size_t(token - (r.prefix_tokens / 4) * 4) * 128 + d]
            : p.raw_keys[std::size_t(token - r.prefix_tokens) * 128 + d];
        const float value = detail::roundRepresentation(raw, p.representation);
        pooled = member == 0 ? value : pooled + value;
    }
    __shared__ float shared[128];
    prepareHead(detail::roundRepresentation(pooled * 0.25f, p.representation), p.key_norm,
                position, p, p.pooled_keys + std::size_t(block) * 128, shared);
}
/** @brief Score eight complete pools with fixed 32-lane index-head reductions. */
__global__ __launch_bounds__(256, 2) void scorePools(QwenSparseIndexer p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    require(!p.transaction_prior || p.transaction_prior->row_capacity==p.row_capacity);
    const int row = int(blockIdx.y);
    if (row >= r.live_rows) return;
    const int blocks = (r.prefix_tokens + row + 1) / 4;
    if (blocks <= QwenSparseBlocks) return;
    const int subgroup = int(threadIdx.x) / 32;
    const int lane = int(threadIdx.x) % 32;
    const int block = int(blockIdx.x) * 8 + subgroup;
    // A whole 32-lane subgroup participates or exits, including on wave64.
    if (block >= blocks) return;
    float score = 0.0f;
    for (int head = 0; head < 4; ++head)
    {
        float value = 0.0f;
        for (int d = lane; d < 128; d += 32)
            value = fmaf(p.prepared_query[(std::size_t(row) * 4 + head) * 128 + d],
                         p.pooled_keys[std::size_t(block) * 128 + d], value);
        for (int distance = 16; distance; distance /= 2) value += shuffleDown(value, distance);
        if (lane == 0) score += fmaxf(0.0f, value);
    }
    if (lane == 0)
    {
        score *= 0.08838834764831844055f;
        require(isfinite(score));
        p.scores[std::size_t(row) * p.block_capacity + block] = score;
    }
}
/** @brief Reduce an exact integer count inside one fixed 32-lane row owner. */
__device__ __forceinline__ int sumCount(int value)
{
    for (int distance = 16; distance; distance /= 2)
    {
#if defined(__HIPCC__)
        value += __shfl_xor(value, distance, 32);
#else
        value += __shfl_xor_sync(0xffffffffu, value, distance, 32);
#endif
    }
    return value;
}
/** @brief Return the predicate mask for this 32-lane owner on warp32 or wave64. */
__device__ __forceinline__ std::uint32_t ballot(bool predicate)
{
#if defined(__HIPCC__)
    const auto mask = __ballot(predicate);
    return static_cast<std::uint32_t>(mask >> ((int(threadIdx.x) & 32) ? 32 : 0));
#else
    return __ballot_sync(0xffffffffu, predicate);
#endif
}
/** @brief Exact deterministic top-512 threshold and sorted sparse-index compaction. */
__global__ __launch_bounds__(256, 2) void selectPools(QwenSparseIndexer p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    require(!p.transaction_prior || p.transaction_prior->row_capacity==p.row_capacity);
    const int row = int(blockIdx.x) * 8 + int(threadIdx.x) / 32;
    const int lane = int(threadIdx.x) % 32;
    if (row >= p.row_capacity) return;
    if (row >= r.live_rows)
    {
        if (lane == 0) p.selected_count[row] = 0;
        return;
    }
    const int visible = r.prefix_tokens + row + 1;
    const int blocks = visible / 4;
    auto *output = p.selected + std::size_t(row) * QwenSparseSelected;
    if (blocks <= QwenSparseBlocks)
    {
        for (int token = lane; token < visible; token += 32) output[token] = token;
        if (lane == 0) p.selected_count[row] = visible;
        return;
    }
    const float *scores = p.scores + std::size_t(row) * p.block_capacity;
    std::uint32_t threshold = 0;
    // Non-negative finite IEEE scores sort by their unsigned bit pattern.
    // The kth-largest threshold is independent of any histogram/atomic order.
    for (int bit = 30; bit >= 0; --bit)
    {
        const std::uint32_t candidate = threshold | (std::uint32_t{1} << bit);
        int local = 0;
        for (int block = lane; block < blocks; block += 32)
            local += bounded_selection::canonicalScoreBits(detail::floatBits(scores[block])) >= candidate;
        if (sumCount(local) >= QwenSparseBlocks) threshold = candidate;
    }
    int greater = 0;
    for (int block = lane; block < blocks; block += 32)
        greater += bounded_selection::canonicalScoreBits(detail::floatBits(scores[block])) > threshold;
    const int tie_budget = QwenSparseBlocks - sumCount(greater);
    int tie_seen = 0;
    int written = 0;
    const std::uint32_t prior_lanes = (std::uint32_t{1} << lane) - 1u;
    for (int base = 0; base < blocks; base += 32)
    {
        const int block = base + lane;
        const auto bits = block < blocks ? bounded_selection::canonicalScoreBits(detail::floatBits(scores[block])) : 0xffffffffu;
        const int equal = block < blocks && bits == threshold;
        const auto equals = ballot(equal);
        const int equal_before = __popc(equals & prior_lanes);
        const int equal_total = __popc(equals);
        const int chosen = block < blocks && (bits > threshold || (equal && tie_seen + equal_before < tie_budget));
        const auto chosen_mask = ballot(chosen);
        const int index = written + __popc(chosen_mask & prior_lanes);
        const int chunk_written = __popc(chosen_mask);
        if (chosen)
            for (int member = 0; member < 4; ++member) output[index * 4 + member] = block * 4 + member;
        written += chunk_written;
        tie_seen += equal_total;
    }
    require(written == QwenSparseBlocks);
    if (lane < visible % 4) output[2048 + lane] = blocks * 4 + lane;
    if (lane == 0) p.selected_count[row] = 2048 + visible % 4;
}
/** @brief Exact CTA-wide count using two publication barriers and eight warp totals. */
__device__ __forceinline__ int sumWide(int local, int *warps)
{
    const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
    local = sumCount(local);
    if (lane == 0) warps[warp] = local;
    __syncthreads();
    int sum = 0;
    for (int i = 0; i < 8; ++i) sum += warps[i];
    __syncthreads();
    return sum;
}
/**
 * @brief Small-M top-k with eight-way context parallelism and exact integer ordering.
 *
 * Score arithmetic is untouched. Only order-independent integer comparisons
 * and population counts move between threads. This keeps a long-context M=1
 * transaction from scanning every score thirty-one times on one half-wave.
 */
/** @brief Exact lexical storage for the Installed CTA-wide integer selector. */
struct WideSelectionStorage final
{
    int counts[8], equals[8], chosen[8];
};
static_assert(sizeof(WideSelectionStorage) == 96);
/**
 * @brief Execute the Installed Wide row program on all 256 threads of its CTA.
 * @param p Original native bindings and immutable arithmetic policy.
 * @param r Authenticated device-owned replay; never a host-derived live frontier.
 * @param row Unique physical row owned by this CTA.
 * @param storage Exclusive lexical shared storage, initialized at each publication edge.
 */
__device__ __forceinline__ void selectPoolsWideRow(const QwenSparseIndexer& p,
    const QwenSparseReplay& r, int row, WideSelectionStorage& storage)
{
    const int thread = int(threadIdx.x), lane = thread % 32, warp = thread / 32;
    if (row >= r.live_rows)
    {
        if (thread == 0) p.selected_count[row] = 0;
        return;
    }
    const int visible = r.prefix_tokens + row + 1, blocks = visible / 4;
    auto *output = p.selected + std::size_t(row) * QwenSparseSelected;
    if (blocks <= QwenSparseBlocks)
    {
        for (int token = thread; token < visible; token += 256) output[token] = token;
        if (thread == 0) p.selected_count[row] = visible;
        return;
    }
    const auto *scores = p.scores + std::size_t(row) * p.block_capacity;
    auto* counts = storage.counts;
    auto* equals = storage.equals;
    auto* chosen = storage.chosen;
    std::uint32_t threshold = 0;
    for (int bit = 30; bit >= 0; --bit)
    {
        const auto candidate = threshold | (std::uint32_t{1} << bit);
        int count = 0;
        for (int block = thread; block < blocks; block += 256) count += bounded_selection::canonicalScoreBits(detail::floatBits(scores[block])) >= candidate;
        if (sumWide(count, counts) >= QwenSparseBlocks) threshold = candidate;
    }
    int greater = 0;
    for (int block = thread; block < blocks; block += 256) greater += bounded_selection::canonicalScoreBits(detail::floatBits(scores[block])) > threshold;
    const int tie_budget = QwenSparseBlocks - sumWide(greater, counts);
    int tie_seen = 0, written = 0;
    const auto prior_lanes = (std::uint32_t{1} << lane) - 1u;
    for (int base = 0; base < blocks; base += 256)
    {
        const int block = base + thread;
        const auto bits = block < blocks ? bounded_selection::canonicalScoreBits(detail::floatBits(scores[block])) : 0xffffffffu;
        const bool equal = block < blocks && bits == threshold;
        const auto equal_mask = ballot(equal);
        if (lane == 0) equals[warp] = __popc(equal_mask);
        __syncthreads();
        int equal_before = __popc(equal_mask & prior_lanes);
        for (int i = 0; i < warp; ++i) equal_before += equals[i];
        const bool keep = block < blocks && (bits > threshold || (equal && tie_seen + equal_before < tie_budget));
        const auto kept_mask = ballot(keep);
        if (lane == 0) chosen[warp] = __popc(kept_mask);
        __syncthreads();
        int index = written + __popc(kept_mask & prior_lanes);
        for (int i = 0; i < warp; ++i) index += chosen[i];
        if (keep)
            for (int member = 0; member < 4; ++member) output[index * 4 + member] = block * 4 + member;
        for (int i = 0; i < 8; ++i) { tie_seen += equals[i]; written += chosen[i]; }
        __syncthreads();
    }
    require(written == QwenSparseBlocks);
    if (thread < visible % 4) output[2048 + thread] = blocks * 4 + thread;
    if (thread == 0) p.selected_count[row] = 2048 + visible % 4;
}
/** @brief Original Installed Wide entrypoint; geometry and static storage remain explicit. */
__global__ __launch_bounds__(256, 2) void selectPoolsWide(QwenSparseIndexer p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    require(!p.transaction_prior || p.transaction_prior->row_capacity==p.row_capacity);
    __shared__ WideSelectionStorage storage;
    selectPoolsWideRow(p, r, int(blockIdx.x), storage);
}
/** @brief Fixed32-lane unsigned key reduction, independent of wave32/wave64 scheduling. */
__device__ __forceinline__ std::uint64_t reduceKey(std::uint64_t value, bool minimum)
{
    for (int distance = 16; distance; distance /= 2)
    {
#if defined(__HIPCC__)
        const auto other = __shfl_xor(value, distance, 32);
#else
        const auto other = __shfl_xor_sync(0xffffffffu, value, distance, 32);
#endif
        value = minimum ? (other < value ? other : value) : (other > value ? other : value);
    }
    return value;
}

/**
 * @brief Publish the exact descending histogram pivot from one fixed 32-lane owner.
 * @param storage Exclusive CTA bank after its histogram-ready publication barrier.
 * @details Only the first logical group of 32 threads calls this helper, including
 * the lower half of a HIP wave64. Each lane owns eight contiguous integer bins.
 * A five-step suffix of lane totals locates the unique owning interval; its lane
 * then scans at most eight bins. This adapts the pinned DeepSelect algorithm in
 * common_parts.cuh::find_pivot_in_histogram (bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18).
 * No vendor PTX is used. Histogram generation remains unchanged, so every count
 * and their total are bounded by the admitted bank capacity, at most 2560 keys.
 * The owner ballot consumes every lane's old rank before any state publication.
 * There is no CTA barrier here: all 256 threads join the caller's final barrier.
 */
template<int Block, int Slack>
__device__ __forceinline__ void selectBoundedHistogramPivot(
    bounded_selection::Storage<Block, Slack> &storage)
{
    const int lane = int(threadIdx.x) % 32;
    const std::uint32_t old_rank = static_cast<std::uint32_t>(storage.rank);
    const int old_shift = storage.shift;
    const std::uint64_t old_prefix = storage.prefix;
    std::uint32_t counts[8];
    std::uint32_t local_total = 0;
    #pragma unroll
    for (int index = 0; index < 8; ++index)
    {
        counts[index] = storage.histogram[lane * 8 + index];
        local_total += counts[index];
    }
    std::uint32_t inclusive_suffix = local_total;
    #pragma unroll
    for (int distance = 1; distance <= 16; distance *= 2)
    {
#if defined(__HIPCC__)
        const auto other = __shfl_down(inclusive_suffix, distance, 32);
#else
        const auto other = __shfl_down_sync(0xffffffffu, inclusive_suffix, distance, 32);
#endif
        if (lane + distance < 32) inclusive_suffix += other;
    }
#if defined(__HIPCC__)
    const auto parent_total = __shfl(inclusive_suffix, 0, 32);
#else
    const auto parent_total = __shfl_sync(0xffffffffu, inclusive_suffix, 0, 32);
#endif
    require(old_rank >= 1u && old_rank <= parent_total &&
        parent_total <= static_cast<std::uint32_t>(storage.size));
    std::uint32_t greater = inclusive_suffix - local_total;
    const bool owns = greater < old_rank && old_rank <= inclusive_suffix;
    // CUDA's synchronized ballot and HIP's active logical half-wave consume all
    // interval predicates before the sole publisher can change shared rank.
    const auto owners = ballot(owns);
    require(__popc(owners) == 1);
    if (owns)
    {
        int selected_index = -1;
        std::uint32_t residual = 0, population = 0;
        // Constant unrolled accesses keep the eight counts register-addressable;
        // a dynamic counts[selected_index] could introduce private-memory traffic.
        #pragma unroll
        for (int index = 7; index >= 0; --index)
        {
            if (selected_index < 0)
            {
                if (old_rank <= greater + counts[index])
                {
                    selected_index = index;
                    residual = old_rank - greater;
                    population = counts[index];
                }
                else greater += counts[index];
            }
        }
        require(selected_index >= 0 && residual >= 1u && residual <= population);
        ++storage.observation.radix_passes;
        storage.prefix = old_prefix | (std::uint64_t(lane * 8 + selected_index) << old_shift);
        storage.rank = static_cast<int>(residual);
        storage.done = residual == population || old_shift == 32;
        if (!storage.done) storage.shift = old_shift - 8;
    }
}

/**
 * @brief Compact only a bounded shared bank, preserving original ascending IDs.
 * @param storage Exclusive CTA lexical storage; all256 threads must participate.
 * @details Only the four score bytes enter radix; stable ascending original-ID
 * order resolves the remaining cutoff tie. Equal scores need no histogram.
 * Histogram slots 0 through 7 become warp tie counts only after radix retires;
 * shift becomes the cumulative tie count after that same CTA publication edge.
 * The retained threshold remains the true minimum complete 64-bit key.
 */
template<int Block, int Slack>
__device__ __forceinline__ void compactBounded(bounded_selection::Storage<Block, Slack> &storage)
{
    const int thread = int(threadIdx.x), lane = thread % 32, warp = thread / 32;
    require(storage.size >= QwenSparseBlocks && storage.size <= storage.Capacity);
    const auto *source = storage.keys[storage.active];
    auto *target = storage.keys[storage.active ^ 1];
    std::uint64_t minimum = UINT64_MAX, maximum = 0;
    for (int i = thread; i < storage.size; i += 256)
    {
        const auto value = source[i];
        minimum = value < minimum ? value : minimum;
        maximum = value > maximum ? value : maximum;
    }
    minimum = reduceKey(minimum, true);
    maximum = reduceKey(maximum, false);
    if (lane == 0) { storage.warp_min[warp] = minimum; storage.warp_max[warp] = maximum; }
    __syncthreads();
    if (thread == 0)
    {
        minimum = UINT64_MAX; maximum = 0;
        for (int i = 0; i < 8; ++i)
        {
            minimum = storage.warp_min[i] < minimum ? storage.warp_min[i] : minimum;
            maximum = storage.warp_max[i] > maximum ? storage.warp_max[i] : maximum;
        }
        storage.shift = bounded_selection::firstShift(minimum, maximum);
        storage.prefix = storage.shift < 32 ? minimum & 0xffffffff00000000ULL :
            maximum & bounded_selection::prefixMask(storage.shift);
        storage.rank = QwenSparseBlocks;
        storage.done = storage.shift < 32;
    }
    __syncthreads();
    while (!storage.done)
    {
        storage.histogram[thread] = 0;
        __syncthreads();
        const auto mask = bounded_selection::prefixMask(storage.shift);
        const auto prefix = storage.prefix;
        for (int i = thread; i < storage.size; i += 256)
        {
            const auto value = source[i];
            if ((value & mask) == prefix)
                atomicAdd(&storage.histogram[(value >> storage.shift) & 255u], 1u);
        }
        __syncthreads();
        // One logical warp selects the pivot while all CTA participants retain
        // the same existing histogram-ready and final publication barriers.
        if (thread < 32) selectBoundedHistogramPivot(storage);
        __syncthreads();
    }
    // All histogram readers retired at the preceding radix publication edge.
    // Reuse shift for tie_seen; rank is the residual cutoff-bucket budget.
    if (thread == 0) { storage.written = 0; storage.pivot = storage.prefix; storage.shift = 0; }
    __syncthreads();
    minimum = UINT64_MAX;
    const auto prior_lanes = (std::uint32_t{1} << lane) - 1u;
    for (int base = 0; base < storage.size; base += 256)
    {
        const int index = base + thread;
        const auto value = index < storage.size ? source[index] : 0;
        const auto score = value & 0xffffffff00000000ULL;
        const bool equal = index < storage.size && score == storage.pivot;
        const auto equal_mask = ballot(equal);
        if (lane == 0) storage.histogram[warp] = __popc(equal_mask);
        __syncthreads();
        int equal_before = __popc(equal_mask & prior_lanes), equal_total = 0;
        for (int i = 0; i < 8; ++i)
        {
            if (i < warp) equal_before += storage.histogram[i];
            equal_total += storage.histogram[i];
        }
        const bool keep = index < storage.size && (score > storage.pivot ||
            (equal && storage.shift + equal_before < storage.rank));
        const auto mask = ballot(keep);
        if (lane == 0) storage.warp_counts[warp] = __popc(mask);
        __syncthreads();
        int before = __popc(mask & prior_lanes), total = 0;
        for (int i = 0; i < 8; ++i)
        {
            if (i < warp) before += storage.warp_counts[i];
            total += storage.warp_counts[i];
        }
        if (keep)
        {
            require(storage.written + before < QwenSparseBlocks);
            target[storage.written + before] = value;
            minimum = value < minimum ? value : minimum;
        }
        __syncthreads();
        if (thread == 0) { storage.written += total; storage.shift += equal_total; }
        __syncthreads();
    }
    minimum = reduceKey(minimum, true);
    if (lane == 0) storage.warp_min[warp] = minimum;
    __syncthreads();
    if (thread == 0)
    {
        minimum = UINT64_MAX;
        for (int i = 0; i < 8; ++i)
            minimum = storage.warp_min[i] < minimum ? storage.warp_min[i] : minimum;
        require(storage.written == QwenSparseBlocks && minimum >= storage.threshold);
        ++storage.observation.compactions;
        storage.observation.compacted_keys += static_cast<std::uint64_t>(storage.size);
        storage.threshold = minimum;
        storage.size = QwenSparseBlocks;
        storage.active ^= 1;
    }
    __syncthreads();
}

/**
 * @brief Read each live score once, filter, and periodically compact bounded keys.
 * @details One CTA owns one original row on CUDA/HIP. Traversal is sequential:
 * there is no upstream randomized expected-work guarantee. Score arithmetic,
 * complete-pool visibility, canonical ties, tails and inactive storage are fixed.
 */
/**
 * @brief Execute one exact bounded scan/filter/compact row on all 256 CTA threads.
 * @param p Original native inputs and exclusive selection outputs.
 * @param r Authenticated execution-owned replay.
 * @param row Unique physical row owned by this CTA.
 * @param storage Exclusive initialized-lifetime lexical storage, with dirty payload permitted.
 */
template<int Block, int Slack>
__device__ __forceinline__ void selectPoolsBoundedRow(const QwenSparseIndexer& p,
    const QwenSparseReplay& r, int row, bounded_selection::Storage<Block, Slack>& storage)
{
    const int thread = int(threadIdx.x), lane = thread % 32, warp = thread / 32;
    if (row >= r.live_rows)
    {
        if (thread == 0)
        {
            p.selected_count[row] = 0;
            if (p.selection_observations) p.selection_observations[row] = {0, 0, 0, 0};
        }
        return;
    }
    const int visible = r.prefix_tokens + row + 1, blocks = visible / QwenSparsePool;
    auto *output = p.selected + std::size_t(row) * QwenSparseSelected;
    if (blocks <= QwenSparseBlocks)
    {
        for (int token = thread; token < visible; token += 256) output[token] = token;
        if (thread == 0)
        {
            p.selected_count[row] = visible;
            if (p.selection_observations) p.selection_observations[row] = {0, 0, 0, 0};
        }
        return;
    }
    if (thread == 0) bounded_selection::begin(storage);
    __syncthreads();
    const auto *scores = p.scores + std::size_t(row) * p.block_capacity;
    const auto prior_lanes = (std::uint32_t{1} << lane) - 1u;
    for (int block_base = 0; block_base < blocks; block_base += Block)
    {
        const int block_end = blocks - block_base < Block ? blocks : block_base + Block;
        for (int base = block_base; base < block_end; base += 256)
        {
            const int id = base + thread;
            std::uint64_t value = 0;
            if (id < block_end)
            {
                const auto bits = detail::floatBits(scores[id]);
                require(bounded_selection::validScoreBits(bits));
                value = bounded_selection::key(bits, static_cast<std::uint32_t>(id));
            }
            const bool keep = id < block_end && value > storage.threshold;
            const auto mask = ballot(keep);
            if (lane == 0) storage.warp_counts[warp] = __popc(mask);
            __syncthreads();
            int before = __popc(mask & prior_lanes), total = 0;
            for (int i = 0; i < 8; ++i)
            {
                if (i < warp) before += storage.warp_counts[i];
                total += storage.warp_counts[i];
            }
            if (keep)
            {
                require(storage.size + before < storage.Capacity);
                storage.keys[storage.active][storage.size + before] = value;
            }
            __syncthreads();
            if (thread == 0)
            {
                storage.size += total;
                storage.observation.score_elements += static_cast<std::uint64_t>(
                    block_end - base < 256 ? block_end - base : 256);
            }
            __syncthreads();
        }
        if (storage.size >= QwenSparseBlocks + Slack) compactBounded(storage);
    }
    if (storage.size > QwenSparseBlocks) compactBounded(storage);
    require(storage.size == QwenSparseBlocks);
    for (int i = thread; i < QwenSparseBlocks; i += 256)
    {
        const int id = bounded_selection::originalId(storage.keys[storage.active][i]);
        for (int member = 0; member < QwenSparsePool; ++member) output[i * QwenSparsePool + member] = id * QwenSparsePool + member;
    }
    if (thread < visible % QwenSparsePool) output[2048 + thread] = blocks * QwenSparsePool + thread;
    if (thread == 0)
    {
        p.selected_count[row] = 2048 + visible % QwenSparsePool;
        if (p.selection_observations) p.selection_observations[row] = storage.observation;
    }
}
/** @brief Original explicit bounded entrypoint; constant geometry and storage are unchanged. */
template<int Block, int Slack>
__global__ __launch_bounds__(256, 1) void selectPoolsBounded(QwenSparseIndexer p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    require(!p.transaction_prior || p.transaction_prior->row_capacity == p.row_capacity);
    __shared__ bounded_selection::Storage<Block, Slack> storage;
    selectPoolsBoundedRow(p, r, int(blockIdx.x), storage);
}
/**
 * @brief Rotate TQ queries once into the caller's not-yet-published output row.
 *
 * Every partition reads this row before the final reducer replaces it. Warp-
 * coalesced matrix rows preserve bandwidth without repeating O(D²) work in
 * each selected-token partition or allocating a second query workspace.
 */
__global__ __launch_bounds__(256, 2) void rotateQueries(QwenSparseAttention p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    const int row = int(blockIdx.x), head = int(blockIdx.y), d = int(threadIdx.x);
    if (row >= r.live_rows) return;
    const int kv_head = (p.query_head_start + head) / 12 - p.kv_head_start;
    const auto *rotation = p.kv.keys.rotations[kv_head];
    const auto offset = (std::size_t(row) * p.query_heads + head) * 256;
    __shared__ float query[256];
    query[d] = p.query[offset + d];
    __syncthreads();
    const int subgroup = d / 32, lane = d % 32;
    for (int output = subgroup; output < 256; output += 8)
    {
        float sum = 0.0f;
        for (int coordinate = lane; coordinate < 256; coordinate += 32)
            sum = fmaf(rotation[output * 256 + coordinate], query[coordinate], sum);
        for (int distance = 16; distance; distance /= 2) sum += shuffleDown(sum, distance);
        if (lane == 0) p.output[offset + output] = sum;
    }
}
/**
 * @brief Compute one selected native-KV partition in a fixed arithmetic order.
 * @tparam KeyCodec Immutable native key encoding selected before capture.
 * @tparam ValueCodec Immutable native value encoding selected before capture.
 * @tparam TransformKeys Whether the declared key bank requires selected-only RoPE.
 * @param p Persistent storage and device-owned replay and selection bindings.
 * @details Each CTA owns one row/head/partition. Every QK lane retains its eight
 * query coordinates across selected tokens, avoiding repeated shared loads.
 * QK coordinates, softmax reductions and PV tokens retain their original order.
 * Empty partitions publish metadata only, and inactive rows publish nothing.
 */
template<TensorType KeyCodec, TensorType ValueCodec, bool TransformKeys>
__global__ __launch_bounds__(256, 2) void selectedAttentionPartitions(QwenSparseAttention p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    const int row = int(blockIdx.x);
    const int head = int(blockIdx.y);
    const int d = int(threadIdx.x);
    const int kv_head = (p.query_head_start + head) / 12 - p.kv_head_start;
    auto &partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * QwenSparsePartitions + blockIdx.z];
    float *output = partial.weighted;
    if (row >= r.live_rows) return; // Spare query capacity owns no native reads or partial writes.
    const int total_count = p.selected_count[row];
    require(total_count > 0 && total_count <= QwenSparseSelected);
    const int begin = int(blockIdx.z) * QwenSparsePartitionTokens;
    const int count = max(0, min(QwenSparsePartitionTokens, total_count - begin));
    if (count == 0)
    {
        // Only metadata is live for an empty arithmetic partition. The fixed
        // reducer substitutes its mathematical +0 without reading stale payload.
        if (d == 0)
        {
            partial.maximum = -INFINITY; partial.denominator = 0.0f;
            partial.key_elements = partial.value_elements = 0;
        }
        return;
    }
    const auto *selected = p.selected + std::size_t(row) * QwenSparseSelected + begin;
    const auto *query = p.query + (std::size_t(row) * p.query_heads + head) * 256;
    int origin = 0, capacity = 0;
    require(detail::resolveMapping(p.kv, r, row, p.token_capacity, origin, capacity));
    if (TransformKeys && p.kv.key_rotary_source==QwenSparseRotarySource::TablePlanes)
        require(static_cast<long long>(r.position_origin) + r.prefix_tokens + row < p.kv.key_position_capacity);
    __shared__ float logits[QwenSparsePartitionTokens];
    __shared__ float reduction[256];
    __shared__ float shared_query[256];
    if constexpr (KeyCodec == TensorType::TQ4 || KeyCodec == TensorType::TQ8)
        shared_query[d] = p.output[(std::size_t(row) * p.query_heads + head) * 256 + d];
    else shared_query[d] = query[d];
    for (int index = d; index < count; index += 256)
    {
        const int token = selected[index];
        require(token >= 0 && token < r.prefix_tokens + row + 1 &&
                (begin + index == 0 || token > selected[index - 1]));
    }
    __syncthreads();
    const int subgroup = d / 32;
    const int lane = d % 32;
    // The query is immutable throughout this partition. Make each lane's query
    // lifetime explicit across the selected-token loop rather than relying on
    // compiler load hoisting. These are the same FP32 values, with no new
    // arithmetic or workspace; native resource and timing gates check the trade.
    float lane_query[QwenSparseHeadDim / 32];
#pragma unroll
    for (int step = 0; step < QwenSparseHeadDim / 32; ++step)
        lane_query[step] = shared_query[lane + step * 32];
    for (int index = subgroup; index < count; index += 8)
    {
        const int logical = selected[index];
        const int physical = detail::physicalRow(logical, origin, capacity);
        float score = 0.0f;
        // Each lane owns exactly eight ascending coordinates. Express that
        // fixed trip count so unrelated observation code cannot make LLVM roll
        // this loop into serialized key/query loads. The dependent FMA chain
        // retains the original order; only independent loads can move ahead.
#pragma unroll
        for (int step = 0; step < QwenSparseHeadDim / 32; ++step)
        {
            const int coordinate = lane + step * 32;
            // The immutable cache encoding is selected before capture. Native
            // post-RoPE banks must not pay registers for unused sine/exp code.
            float key;
            if constexpr(TransformKeys)key=detail::readKeyAs<KeyCodec>(p.kv,physical,logical,kv_head,coordinate,r.position_origin);
            else key=detail::readBankAs<KeyCodec,detail::BankRole::Key>(p.kv.keys,physical,kv_head,coordinate);
            score=fmaf(lane_query[step],key,score);
        }
        for (int distance = 16; distance; distance /= 2) score += shuffleDown(score, distance);
        if (lane == 0) logits[index] = score * 0.0625f;
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int index = d; index < count; index += 256) maximum = fmaxf(maximum, logits[index]);
    reduction[d] = maximum;
    __syncthreads();
    for (int width = 128; width; width /= 2)
    {
        if (d < width) reduction[d] = fmaxf(reduction[d], reduction[d + width]);
        __syncthreads();
    }
    maximum = reduction[0];
    float sum = 0.0f;
    for (int index = d; index < count; index += 256)
    {
        logits[index] = expf(logits[index] - maximum);
        sum += logits[index];
    }
    __syncthreads();
    reduction[d] = sum;
    __syncthreads();
    for (int width = 128; width; width /= 2)
    {
        if (d < width) reduction[d] += reduction[d + width];
        __syncthreads();
    }
    const float denominator = reduction[0];
    require(isfinite(denominator) && denominator > 0.0f);
    float result = 0.0f;
    for (int index = 0; index < count; ++index)
    {
        const int physical = detail::physicalRow(selected[index], origin, capacity);
        result = fmaf(logits[index],
                      detail::readBankAs<ValueCodec>(p.kv.values, physical, kv_head, d), result);
    }
    output[d] = result;
    if (d == 0) { partial.maximum = maximum; partial.denominator = denominator; }
    if (d == 0 && p.observations)
    {
        // These are logical decoded coordinates, not physical transactions or
        // codec bytes: every selected key contributes one coordinate per QK
        // dimension and pre-RoPE coordinates additionally decode their partner.
        // Each PV dimension decodes every selected value once. The unique
        // metadata writer derives exact bounded integer totals without touching
        // the FP32 reduction tree or synchronizing diagnostic-only work.
        partial.key_elements = static_cast<std::uint32_t>(count *
            (256 + (TransformKeys ? p.kv.key_rotary_dim : 0)));
        partial.value_elements = static_cast<std::uint32_t>(count * 256);
    }
}
/**
 * @brief Broadcast one FP32 operand within an active logical 32-lane subgroup.
 * @param value Source lane's original FP32 bit pattern, without conversion.
 * @param source Subgroup-local lane index in[0,31], fixed by the fold iteration.
 * @return The selected source lane's FP32 operand in every participating lane.
 * @pre Every lane of the aligned logical subgroup participates in this call.
 * @details CUDA uses a complete warp mask. HIP width32 confines wave64 reads to
 * the active lower half-wave. Callers consume register operands, not shared
 * publications; this helper introduces no barrier or floating-point operation.
 */
__device__ __forceinline__ float broadcastAttentionLane32(float value,int source)
{
#if defined(__HIPCC__)
    return __shfl(value,source,32);
#else
    return __shfl_sync(0xffffffffu,value,source,32);
#endif
}
/**
 * @brief Borrow one reducer CTA's shared metadata banks without owning storage.
 * @details The kernel owns all four fixed-capacity banks once. Every runtime
 * fold branch borrows the same arrays; no helper specialization allocates an
 * independent bank. Only its FoldParts prefix is published and consumed.
 * Production 8 never accesses these banks after compile-time specialization.
 */
struct QwenSparsePartialMetadataView
{
    float (&maximum)[33]; ///< Bit-exact cached original partial maxima.
    float (&denominator)[33]; ///< Bit-exact cached original partial denominators.
    std::uint32_t (&keys)[33]; ///< Original counters, authored only when observed.
    std::uint32_t (&values)[33]; ///< Original counters, authored only when observed.
};
/**
 * @brief Fold a fixed structural prefix of its admitted partial storage in order.
 * @tparam FoldParts Closed compile-time prefix bound: 8, 16, 32 or 33 records.
 * @tparam StorageParts Original arithmetic/storage program: production 8 or test 33.
 * @param p Original read-only replay/selection and output-observation bindings.
 * @param parts Original row/head partial base with the StorageParts record stride.
 * @param scales Exclusive CTA shared storage for the selected prefix scales.
 * @param denominator Exclusive CTA shared scalar published by output owner zero.
 * @param row Original request-local row, used only for observation ownership.
 * @param head Original participant-local query head.
 * @param d Original output-coordinate owner in this 256-thread CTA.
 * @param metadata Borrowed kernel-owned shared banks, used only by StorageParts33.
 * @return Numerator after the ordered prefix and exact finite empty-tail effect.
 * @details Every CTA thread invokes the same specialization. Storage8 retains
 * its original owner-zero metadata fold. Storage33 distributes exponentials
 * across one logical subgroup while every member retains the ordered maximum
 * and denominator-FMA sequence; only owner zero publishes observations. The
 * existing publication barrier precedes all weighted loads. The caller
 * derives this bound from the immutable-in-flight device selected count, so
 * omitted records are producer-authored empty virtual tiles. Production 8
 * omits no records. At most two zero-product
 * FMAs preserve their finite RN fixed point even with output-only FP32 flush.
 */
template<int FoldParts,int StorageParts>
__device__ __forceinline__ float foldAttentionPrefix(
    const QwenSparseAttention& p,const QwenSparseAttentionPartial* parts,
    float* scales,float& denominator,int row,int head,int d,
    QwenSparsePartialMetadataView metadata)
{
    static_assert((StorageParts==8 && FoldParts==8) ||
                  (StorageParts==33 && (FoldParts==8 || FoldParts==16 || FoldParts==32 || FoldParts==33)),
                  "Closed attention fold/storage geometry");
    if constexpr (StorageParts == 33)
    {
        if (d < FoldParts)
        {
            metadata.maximum[d] = parts[d].maximum;
            metadata.denominator[d] = parts[d].denominator;
            if (p.observations)
            {
                metadata.keys[d] = parts[d].key_elements;
                metadata.values[d] = parts[d].value_elements;
            }
        }
        // One writer per cached scalar; all CTA readers wait for publication.
        // Unused capacity and unobserved counters remain unauthored/unread.
        __syncthreads();
    }
    if constexpr (StorageParts == 33)
    {
        // The first logical subgroup is complete on CUDA warp32 and the lower
        // half of HIP wave64. All32 lanes reach every register broadcast below.
        if (d < 32)
        {
            float maximum = -INFINITY;
            for (int part = 0; part < FoldParts; ++part)
                maximum = fmaxf(maximum, metadata.maximum[part]);
            float localDenominator = 0.0f, localScale = 0.0f;
            if (d < FoldParts)
            {
                localDenominator = metadata.denominator[d];
                localScale = localDenominator == 0.0f ? 0.0f : expf(metadata.maximum[d] - maximum);
                scales[d] = localScale;
            }
            float sum = 0.0f;
            constexpr int SubgroupParts = FoldParts < 32 ? FoldParts : 32;
            for (int part = 0; part < SubgroupParts; ++part)
            {
                const float partDenominator = broadcastAttentionLane32(localDenominator,part);
                const float partScale = broadcastAttentionLane32(localScale,part);
                sum = fmaf(partDenominator,partScale,sum);
            }
            if constexpr (FoldParts == 33)
            {
                // Replicated last-part evaluation appends the same33rd FMA in
                // every lane; only owner0 publishes its shared scale slot.
                const float partDenominator = metadata.denominator[32];
                const float partScale = partDenominator == 0.0f ? 0.0f : expf(metadata.maximum[32] - maximum);
                sum = fmaf(partDenominator,partScale,sum);
                if (d == 0) scales[32] = partScale;
            }
            if (d == 0)
            {
                require(isfinite(sum) && sum > 0.0f);
                denominator = sum;
            }
        }
    }
    if (d == 0)
    {
        if constexpr (StorageParts == 8)
        {
            float maximum = -INFINITY;
            for (int part = 0; part < FoldParts; ++part)
                maximum = fmaxf(maximum, StorageParts == 33 ? metadata.maximum[part] : parts[part].maximum);
            float sum = 0.0f;
            for (int part = 0; part < FoldParts; ++part)
            {
                const float part_denominator = StorageParts == 33 ? metadata.denominator[part] : parts[part].denominator;
                const float part_maximum = StorageParts == 33 ? metadata.maximum[part] : parts[part].maximum;
                scales[part] = part_denominator == 0.0f ? 0.0f : expf(part_maximum - maximum);
                sum = fmaf(part_denominator, scales[part], sum);
            }
            require(isfinite(sum) && sum > 0.0f);
            denominator = sum;
        }
        if (p.observations)
        {
            QwenSparseAttentionObservation observation;
            for (int part = 0; part < FoldParts; ++part)
            {
                observation.key_elements += StorageParts == 33 ? metadata.keys[part] : parts[part].key_elements;
                observation.value_elements += StorageParts == 33 ? metadata.values[part] : parts[part].value_elements;
            }
            observation.selected_tokens = p.selected_count[row];
            p.observations[row * p.query_heads + head] = observation;
        }
    }
    __syncthreads();
    float result = 0.0f;
    for (int part = 0; part < FoldParts; ++part)
    {
        const float part_denominator = StorageParts == 33 ? metadata.denominator[part] : parts[part].denominator;
        const float weighted = part_denominator == 0.0f ? 0.0f : parts[part].weighted[d];
        result = fmaf(weighted, scales[part], result);
    }
    if constexpr (FoldParts < StorageParts) result = fmaf(0.0f, 0.0f, result);
    if constexpr (FoldParts + 1 < StorageParts) result = fmaf(0.0f, 0.0f, result);
    return result;
}
/**
 * @brief Merge the same fixed arithmetic partitions for serial and grouped execution.
 * @tparam ValueCodec Original native value codec and optional output rotation.
 * @tparam Parts Production8 or the explicit FP16 test-native33 arithmetic program.
 * @param p Original row/head bindings and partial storage for that exact program.
 * @details Production 8 keeps its original fold. The explicit 33-part program selects
 * a fixed structural prefix and retains the finite RN effect of its omitted
 * empty tail. This is not universal old 8-part rounding or signed-zero equivalence.
 */
template<TensorType ValueCodec,int Parts=QwenSparsePartitions>
__global__ __launch_bounds__(256, 2) void reduceAttentionPartitions(QwenSparseAttention p)
{
    static_assert(Parts==8 || (Parts==33 && ValueCodec==TensorType::FP16),"Closed attention partial program");
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    const int row = int(blockIdx.x), head = int(blockIdx.y), d = int(threadIdx.x);
    const int kv_head = (p.query_head_start + head) / 12 - p.kv_head_start;
    const auto *parts = p.gpu_partials + (std::size_t(row) * p.query_heads + head) * Parts;
    float *output = p.output + (std::size_t(row) * p.query_heads + head) * 256;
    if (row >= r.live_rows)
    {
        if (d == 0 && p.observations) p.observations[row * p.query_heads + head] = {};
        return;
    }
    __shared__ float scales[Parts];
    __shared__ float denominator;
    __shared__ float rotated_values[256];
    __shared__ float cached_maximum[33], cached_denominator[33];
    __shared__ std::uint32_t cached_keys[33], cached_values[33];
    const QwenSparsePartialMetadataView metadata{
        cached_maximum,cached_denominator,cached_keys,cached_values};
    float result;
    if constexpr (Parts == 33)
    {
        const int selected = p.selected_count[row];
        require(selected > 0 && selected <= QwenSparseSelected);
        // One immutable device count selects a CTA-uniform structural tile bound.
        // The row/head storage stride remains 33 in every specialization.
        if (selected <= 8 * 64)
            result = foldAttentionPrefix<8,33>(p, parts, scales, denominator, row, head, d, metadata);
        else if (selected <= 16 * 64)
            result = foldAttentionPrefix<16,33>(p, parts, scales, denominator, row, head, d, metadata);
        else if (selected <= 32 * 64)
            result = foldAttentionPrefix<32,33>(p, parts, scales, denominator, row, head, d, metadata);
        else
            result = foldAttentionPrefix<33,33>(p, parts, scales, denominator, row, head, d, metadata);
    }
    else
        result = foldAttentionPrefix<8,8>(p, parts, scales, denominator, row, head, d, metadata);
    result /= denominator;
    if constexpr (ValueCodec == TensorType::TQ4 || ValueCodec == TensorType::TQ8)
    {
        rotated_values[d] = result;
        __syncthreads();
        result = 0.0f;
        for (int coordinate = 0; coordinate < 256; ++coordinate)
            result = fmaf(p.kv.values.rotations[kv_head][coordinate * 256 + d], rotated_values[coordinate], result);
    }
    if (p.gate) result *= 1.0f / (1.0f + expf(-p.gate[(std::size_t(row) * p.query_heads + head) * 256 + d]));
    output[d] = result;
}
/** @brief Acceptance-only raw-tail publication, including alias-safe prior-tail reads. */
__global__ __launch_bounds__(128, 2) void commitPartial(QwenSparsePartialCommit p)
{
    const auto r = replay(p.replay,p.acceptance.candidates().rowsPerRequest(),p.transaction_prior ? p.token_capacity : 0x7fffffff);
    require(!p.transaction_prior || p.transaction_prior->row_capacity==p.acceptance.candidates().rowsPerRequest());
    const int accepted=p.acceptance.resolveNative(0);
    require(accepted >= 0 && accepted <= r.live_rows);
    if(r.live_rows==0 || (accepted==0 && !p.transaction_prior))return;
    const int total = r.prefix_tokens + accepted;
    const int begin = total / 4 * 4;
    const int d = int(threadIdx.x);
    float values[3];
    for (int member = 0; member < total % 4; ++member)
    {
        const int token = begin + member;
        values[member] = token < r.prefix_tokens
            ? p.prior_partial[std::size_t(token - (r.prefix_tokens / 4) * 4) * 128 + d]
            : p.raw_keys[std::size_t(token - r.prefix_tokens) * 128 + d];
    }
    __syncthreads();
    for (int member = 0; member < total % 4; ++member) p.partial_keys[member * 128 + d] = values[member];
    if(d==0 && p.work_counters.bound())
        require(addNativeCounter64(p.work_counters.metric(QwenSparseWorkCounter::AcceptedPayload),std::uint64_t(total%4)*128*sizeof(float)));
}
/** @brief Resolve a native value codec at capture time, never inside the key/value loop. */
template<TensorType KeyCodec, class Launch>
void dispatchValue(TensorType value, Launch &&launch)
{
    switch (value)
    {
    case TensorType::FP32: launch.template operator()<KeyCodec, TensorType::FP32>(); return;
    case TensorType::FP16: launch.template operator()<KeyCodec, TensorType::FP16>(); return;
    case TensorType::BF16: launch.template operator()<KeyCodec, TensorType::BF16>(); return;
    case TensorType::Q8_1: launch.template operator()<KeyCodec, TensorType::Q8_1>(); return;
    case TensorType::Q16_1: launch.template operator()<KeyCodec, TensorType::Q16_1>(); return;
    case TensorType::TQ4: launch.template operator()<KeyCodec, TensorType::TQ4>(); return;
    case TensorType::TQ8: launch.template operator()<KeyCodec, TensorType::TQ8>(); return;
    default: throw std::invalid_argument("Qwen sparse attention: native value codec was not admitted");
    }
}
/** @brief Resolve both immutable codecs to a first-class compiled sparse reader. */
template<class Launch>
void dispatchAttention(const QwenSparseKV &kv, Launch &&launch)
{
    switch (kv.keys.codec)
    {
    case TensorType::FP32: dispatchValue<TensorType::FP32>(kv.values.codec, launch); return;
    case TensorType::FP16: dispatchValue<TensorType::FP16>(kv.values.codec, launch); return;
    case TensorType::BF16: dispatchValue<TensorType::BF16>(kv.values.codec, launch); return;
    case TensorType::Q8_1: dispatchValue<TensorType::Q8_1>(kv.values.codec, launch); return;
    case TensorType::Q16_1: dispatchValue<TensorType::Q16_1>(kv.values.codec, launch); return;
    case TensorType::AQ8: dispatchValue<TensorType::AQ8>(kv.values.codec, launch); return;
    case TensorType::TQ4: dispatchValue<TensorType::TQ4>(kv.values.codec, launch); return;
    case TensorType::TQ8: dispatchValue<TensorType::TQ8>(kv.values.codec, launch); return;
    default: throw std::invalid_argument("Qwen sparse attention: native key codec was not admitted");
    }
}
} // namespace llaminar2::qwen4exp::gpu_detail

#undef QSA_GRID_CONSTANT
