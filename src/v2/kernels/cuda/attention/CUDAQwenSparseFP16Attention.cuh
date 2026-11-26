/**
 * @file CUDAQwenSparseFP16Attention.cuh
 * @brief Native CUDA FP16 tiled sparse attention with FP32 online totals.
 *
 * Mirrors ROCmQwenSparseFP16Attention.cuh for backend symmetry on CUDA:
 * - 8-part program (prefill & uniform) shares each 64-token KV tile across 4 heads
 *   (128 threads = 4 warps, 1 warp32 per query head, 2 tokens per lane).
 * - 33-part program (single-request decode & grouped verifier) shares each 64-token
 *   KV tile across 6 heads (192 threads = 6 warps, 1 warp32 per query head) so
 *   every selected tile up to 2051 tokens executes in one single-pass CTA per part.
 *
 * Each warp owns one query head with zero cross-warp softmax or PV barriers:
 * - QK dots reuse each loaded query 128-bit vector across two tokens (lane and
 *   lane + 32) in ascending FP32 FMA order.
 * - Softmax max/sum stay 100% warp-local via 5-step __shfl_xor_sync.
 * - PV splits even and odd token chains into separate half2 accumulators per lane,
 *   widens into FP32 online totals with exact 1/128 probability scaling, and sums
 *   even + odd once at partial publication to preserve the exact parity order.
 * - All storage is caller-owned or static shared memory (~21.5 KB for G6 -> 4 CTAs/SM
 *   on SM86), zero register spills, zero dynamic allocation.
 */
#pragma once

#include <cuda_fp16.h>
#include <cmath>

namespace llaminar2::qwen4exp::cuda_sparse_detail
{
/** @brief Exact online-max rescale multipliers; matches ROCmQwenSparseOnlineRescale.cuh. */
struct SparseHalfOnlineScales
{
    float previous; ///< Scale applied to prior FP32 numerator and denominator totals.
    float tile;     ///< Scale applied to current tile totals.
};

/**
 * @brief Elide only finite zero-difference exponentials while retaining nonfinite behavior.
 * @param maximum Running maximum before this tile.
 * @param tileMaximum Warp-reduced maximum of this tile.
 * @param denominator Running denominator before this tile.
 * @param nextMaximum Caller's fmaxf(maximum, tileMaximum).
 * @return Exact (previous, tile) scale pair.
 */
__device__ __forceinline__ SparseHalfOnlineScales sparseHalfOnlineScales(
    float maximum, float tileMaximum, float denominator, float nextMaximum)
{
    if (isfinite(maximum) && isfinite(tileMaximum))
    {
        if (maximum < tileMaximum)
            return {denominator == 0.0f ? 0.0f : expf(maximum - nextMaximum), 1.0f};
        if (maximum > tileMaximum)
            return {denominator == 0.0f ? 0.0f : 1.0f, expf(tileMaximum - nextMaximum)};
        return {denominator == 0.0f ? 0.0f : 1.0f, 1.0f};
    }
    if (maximum == -INFINITY && denominator == 0.0f && isfinite(tileMaximum))
        return {0.0f, 1.0f};
    return {denominator == 0.0f ? 0.0f : expf(maximum - nextMaximum),
            expf(tileMaximum - nextMaximum)};
}

/** @brief Four naturally aligned packed-half pairs (16 bytes = 8 FP16 values). */
struct alignas(16) SparseHalfVector
{
    half2 pairs[4];
};
static_assert(sizeof(SparseHalfVector) == 16);

/** @brief Two packed-half pairs (8 bytes = 4 FP16 values) for split-half RoPE reads. */
struct alignas(8) SparseHalfDuo
{
    half2 pairs[2];
};
static_assert(sizeof(SparseHalfDuo) == 8);

/**
 * @brief Rotate one native FP16 split-half key pair with exact FP16 rounding boundaries.
 *
 * Uses explicit __fmul_rn / __fadd_rn so --use_fast_math never contracts products
 * across the FP16 rounding boundaries required by the canonical readKey contract.
 */
__device__ __forceinline__ float2 rotateSparseHalfSplitPair(
    const QwenSparseKV& kv, const QwenSparseReplay& r,
    int logical, int coordinate, float low, float high)
{
    float cosine, sine;
    detail::rotaryFactors(kv.key_rotary_source, kv.key_cosine, kv.key_sine,
        r.position_origin + logical, coordinate, 64, kv.key_policy.rope_theta,
        cosine, sine, nullptr, kv.key_negative_log_theta);
    constexpr auto repr = QwenSparseScalar::FP16;
    const float c = detail::roundRepresentation(cosine, repr);
    const float s = detail::roundRepresentation(sine, repr);
    const float first = detail::roundRepresentation(
        __fadd_rn(
            detail::roundRepresentation(__fmul_rn(low, c), repr),
            detail::roundRepresentation(__fmul_rn(-high, s), repr)),
        repr);
    const float second = detail::roundRepresentation(
        __fadd_rn(
            detail::roundRepresentation(__fmul_rn(high, c), repr),
            detail::roundRepresentation(__fmul_rn(low, s), repr)),
        repr);
    return make_float2(first, second);
}

/**
 * @brief Cooperatively load one 64-token x 128-coordinate native FP16 key panel into padded SMEM.
 * @tparam Transform Whether keys require on-read 64-dim split-half RoPE.
 * @tparam Coordinate 0 for the first 128 coordinates, 128 for the second 128 coordinates.
 * @tparam FullPanel True only when the caller proved count == 64.
 * @tparam IdentityAddressing True only when the caller proved sorted identity [begin, begin+count).
 * @tparam Threads CTA thread count (128 for G4 or 192 for G6).
 */
template<bool Transform, int Coordinate, bool FullPanel, bool IdentityAddressing, int Threads>
__device__ __forceinline__ void copySparseHalfKeysPanel(
    const QwenSparseAttention& p, const QwenSparseReplay& r,
    const std::int32_t* selected, int count, int logicalBegin,
    int kvHead, int origin, int capacity, half2* tile)
{
    static_assert(Threads == 128 || Threads == 192, "Closed CUDA cooperative tile-copy geometry");
    for (int vector = int(threadIdx.x); vector < 64 * 16; vector += Threads)
    {
        const int token = vector / 16;
        const int columnVector = vector % 16;
        if constexpr (Transform && Coordinate == 0)
        {
            if (columnVector < 8)
            {
                const int coordinate = columnVector * 4;
                SparseHalfDuo low, high;
#pragma unroll
                for (int item = 0; item < 2; ++item)
                {
                    low.pairs[item] = __float2half2_rn(0.0f);
                    high.pairs[item] = __float2half2_rn(0.0f);
                }
                if (FullPanel || token < count)
                {
                    const int logical = IdentityAddressing ? (logicalBegin + token) : selected[token];
                    const int physical = detail::physicalRow(logical, origin, capacity);
                    const auto* bytes = static_cast<const unsigned char*>(p.kv.keys.data) +
                        std::size_t(physical) * p.kv.keys.token_stride_bytes +
                        std::size_t(kvHead) * p.kv.keys.head_stride_bytes;
                    const auto rawLow = *reinterpret_cast<const SparseHalfDuo*>(bytes + coordinate * sizeof(half));
                    const auto rawHigh = *reinterpret_cast<const SparseHalfDuo*>(bytes + (coordinate + 32) * sizeof(half));
#pragma unroll
                    for (int item = 0; item < 2; ++item)
                    {
                        const float2 x = __half22float2(rawLow.pairs[item]);
                        const float2 y = __half22float2(rawHigh.pairs[item]);
                        const float2 first = rotateSparseHalfSplitPair(p.kv, r, logical, coordinate + item * 2, x.x, y.x);
                        const float2 second = rotateSparseHalfSplitPair(p.kv, r, logical, coordinate + item * 2 + 1, x.y, y.y);
                        low.pairs[item] = __floats2half2_rn(first.x, second.x);
                        high.pairs[item] = __floats2half2_rn(first.y, second.y);
                    }
                }
                *reinterpret_cast<SparseHalfDuo*>(tile + token * 68 + coordinate / 2) = low;
                *reinterpret_cast<SparseHalfDuo*>(tile + token * 68 + coordinate / 2 + 16) = high;
                continue;
            }
        }
        const int pair = columnVector * 4;
        SparseHalfVector value;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            value.pairs[c] = __float2half2_rn(0.0f);
        if (FullPanel || token < count)
        {
            const int logical = IdentityAddressing ? (logicalBegin + token) : selected[token];
            const int physical = detail::physicalRow(logical, origin, capacity);
            const auto* bytes = static_cast<const unsigned char*>(p.kv.keys.data) +
                std::size_t(physical) * p.kv.keys.token_stride_bytes +
                std::size_t(kvHead) * p.kv.keys.head_stride_bytes +
                std::size_t(Coordinate + pair * 2) * sizeof(half);
            value = *reinterpret_cast<const SparseHalfVector*>(bytes);
        }
        *reinterpret_cast<SparseHalfVector*>(tile + token * 68 + pair) = value;
    }
}

/** @brief Dispatch key-panel copy with explicit full-panel and identity permissions. */
template<bool Transform, int Coordinate, bool AllowFullPanel, int Threads>
__device__ __forceinline__ void loadSparseHalfKeys(
    const QwenSparseAttention& p, const QwenSparseReplay& r,
    const std::int32_t* selected, int count, int logicalBegin, bool allVisible,
    int kvHead, int origin, int capacity, half2* tile)
{
    if constexpr (AllowFullPanel)
    {
        if (allVisible)
        {
            if (count == 64)
                copySparseHalfKeysPanel<Transform, Coordinate, true, true, Threads>(
                    p, r, selected, count, logicalBegin, kvHead, origin, capacity, tile);
            else
                copySparseHalfKeysPanel<Transform, Coordinate, false, true, Threads>(
                    p, r, selected, count, logicalBegin, kvHead, origin, capacity, tile);
            return;
        }
        if (count == 64)
        {
            copySparseHalfKeysPanel<Transform, Coordinate, true, false, Threads>(
                p, r, selected, count, logicalBegin, kvHead, origin, capacity, tile);
            return;
        }
    }
    copySparseHalfKeysPanel<Transform, Coordinate, false, false, Threads>(
        p, r, selected, count, logicalBegin, kvHead, origin, capacity, tile);
}

/**
 * @brief Cooperatively load one 32-token x 256-coordinate native FP16 value panel into SMEM.
 * @tparam FullPanel True only when count == 32.
 * @tparam IdentityAddressing True only when sorted identity addressing is proven.
 * @tparam Threads CTA thread count (128 for G4 or 192 for G6).
 */
template<bool FullPanel, bool IdentityAddressing, int Threads>
__device__ __forceinline__ void copySparseHalfValuesPanel(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, int kvHead, int origin, int capacity, half2* tile)
{
    static_assert(Threads == 128 || Threads == 192, "Closed CUDA cooperative tile-copy geometry");
    for (int vector = int(threadIdx.x); vector < 32 * 32; vector += Threads)
    {
        const int token = vector / 32;
        const int pair = (vector % 32) * 4;
        SparseHalfVector value;
#pragma unroll
        for (int c = 0; c < 4; ++c)
            value.pairs[c] = __float2half2_rn(0.0f);
        if (FullPanel || token < count)
        {
            const int logical = IdentityAddressing ? (logicalBegin + token) : selected[token];
            const int physical = detail::physicalRow(logical, origin, capacity);
            const auto* bytes = static_cast<const unsigned char*>(p.kv.values.data) +
                std::size_t(physical) * p.kv.values.token_stride_bytes +
                std::size_t(kvHead) * p.kv.values.head_stride_bytes +
                std::size_t(pair) * sizeof(half2);
            value = *reinterpret_cast<const SparseHalfVector*>(bytes);
        }
        *reinterpret_cast<SparseHalfVector*>(tile + token * 128 + pair) = value;
    }
}

/** @brief Dispatch value-panel copy with explicit full-panel and identity permissions. */
template<bool AllowFullPanel, int Threads>
__device__ __forceinline__ void loadSparseHalfValues(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, bool allVisible, int kvHead, int origin, int capacity, half2* tile)
{
    if constexpr (AllowFullPanel)
    {
        if (allVisible)
        {
            if (count == 32)
                copySparseHalfValuesPanel<true, true, Threads>(
                    p, selected, count, logicalBegin, kvHead, origin, capacity, tile);
            else
                copySparseHalfValuesPanel<false, true, Threads>(
                    p, selected, count, logicalBegin, kvHead, origin, capacity, tile);
            return;
        }
        if (count == 32)
        {
            copySparseHalfValuesPanel<true, false, Threads>(
                p, selected, count, logicalBegin, kvHead, origin, capacity, tile);
            return;
        }
    }
    copySparseHalfValuesPanel<false, false, Threads>(
        p, selected, count, logicalBegin, kvHead, origin, capacity, tile);
}

/**
 * @brief Accumulate 128-coordinate QK dot product for two tokens reusing 128-bit query vectors.
 * @param q_panel Pointer to 64 packed-half pairs (128 coords) of this head's scaled query.
 * @param k0_panel Pointer to 64 packed-half pairs of token (lane).
 * @param k1_panel Pointer to 64 packed-half pairs of token (lane + 32).
 * @param score0 Running FP32 dot product for token (lane).
 * @param score1 Running FP32 dot product for token (lane + 32).
 */
__device__ __forceinline__ void accumulateSparseHalfQKPanel(
    const half2* q_panel,
    const half2* k0_panel,
    const half2* k1_panel,
    float& score0,
    float& score1)
{
#pragma unroll 4
    for (int vec = 0; vec < 16; ++vec)
    {
        const SparseHalfVector qv = *reinterpret_cast<const SparseHalfVector*>(q_panel + vec * 4);
        const SparseHalfVector k0v = *reinterpret_cast<const SparseHalfVector*>(k0_panel + vec * 4);
        const SparseHalfVector k1v = *reinterpret_cast<const SparseHalfVector*>(k1_panel + vec * 4);
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const float2 qf = __half22float2(qv.pairs[c]);
            const float2 k0f = __half22float2(k0v.pairs[c]);
            const float2 k1f = __half22float2(k1v.pairs[c]);
            score0 = fmaf(qf.x, k0f.x, score0);
            score0 = fmaf(qf.y, k0f.y, score0);
            score1 = fmaf(qf.x, k1f.x, score1);
            score1 = fmaf(qf.y, k1f.y, score1);
        }
    }
}

/**
 * @brief Accumulate one 32-token FP16 value panel into separate even and odd parity accumulators.
 * @tparam FullTile True only when count == 64 (omits impossible bounds checks).
 */
template<bool FullTile>
__device__ __forceinline__ void accumulateSparseHalfValuePanel(
    const half* probabilities,
    const half2* tile,
    int chunk,
    int lane,
    int count,
    half2 (&weighted_even)[4],
    half2 (&weighted_odd)[4])
{
#pragma unroll 2
    for (int token = 0; token < 16; ++token)
    {
        const int s_even = 2 * token;
        const int s_odd = s_even + 1;
        if (FullTile || chunk * 32 + s_even < count)
        {
            const half2 p_even = __half2half2(probabilities[chunk * 32 + s_even]);
            const SparseHalfVector v_even = *reinterpret_cast<const SparseHalfVector*>(
                tile + s_even * 128 + lane * 4);
#pragma unroll
            for (int c = 0; c < 4; ++c)
                weighted_even[c] = __hfma2(p_even, v_even.pairs[c], weighted_even[c]);
        }
        if (FullTile || chunk * 32 + s_odd < count)
        {
            const half2 p_odd = __half2half2(probabilities[chunk * 32 + s_odd]);
            const SparseHalfVector v_odd = *reinterpret_cast<const SparseHalfVector*>(
                tile + s_odd * 128 + lane * 4);
#pragma unroll
            for (int c = 0; c < 4; ++c)
                weighted_odd[c] = __hfma2(p_odd, v_odd.pairs[c], weighted_odd[c]);
        }
    }
}

/**
 * @brief Produce fixed batch-invariant FP16 tile streams over native sparse KV on CUDA.
 * @tparam Transform Whether selected keys require the declared 64-dim pre-RoPE transform.
 * @tparam Parts Closed arithmetic stream count: 8 (prefill/uniform) or 33 (single-request decode).
 */
template<bool Transform, int Parts = QwenSparsePartitions>
__global__ __launch_bounds__(32 * (Parts == 33 ? 6 : 4), 2)
void selectedAttentionFP16Tiles(QwenSparseAttention p)
{
    static_assert(Parts == 8 || Parts == 33, "Closed FP16 partition program");
    constexpr int GroupHeads = (Parts == 33) ? 6 : 4;
    constexpr int Threads = 32 * GroupHeads;
    static_assert(12 % GroupHeads == 0, "Attention groups must not cross a KV head");
    static_assert(Parts != 33 || QwenSparseSelected <= Parts * 64,
                  "The single-pass 33 schedule requires at most 33 selected tiles");

    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    const int row = int(blockIdx.x);
    const int wave = int(threadIdx.x) / 32;
    const int lane = int(threadIdx.x) % 32;
    if (row >= r.live_rows) return;

    const int group = (p.query_head_start / GroupHeads + int(blockIdx.y)) * GroupHeads;
    const int head = group + wave - p.query_head_start;
    const bool owned = head >= 0 && head < p.query_heads;
    const int kvHead = group / 12 - p.kv_head_start;
    const int total = p.selected_count[row];
    require(total > 0 && total <= QwenSparseSelected);

    const int visible = r.prefix_tokens + row + 1;
    const bool allVisible = (Parts == 33) && (total == visible);
    const int first = int(blockIdx.z) * 64;
    if (first >= total)
    {
        if (owned && lane == 0)
        {
            auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * Parts + blockIdx.z];
            partial.maximum = -INFINITY;
            partial.denominator = 0.0f;
            partial.key_elements = 0;
            partial.value_elements = 0;
        }
        return;
    }

    int origin = 0, capacity = 0;
    require(detail::resolveMapping(p.kv, r, row, p.token_capacity, origin, capacity));
    if constexpr (Transform)
        if (p.kv.key_rotary_source == QwenSparseRotarySource::TablePlanes)
            require(static_cast<long long>(r.position_origin) + r.prefix_tokens + row < p.kv.key_position_capacity);

    __shared__ __align__(16) half2 queries[GroupHeads][128];
    __shared__ __align__(16) half2 tile[64 * 68];
    __shared__ half probabilities[GroupHeads][64];

#pragma unroll
    for (int part = 0; part < 4; ++part)
    {
        const int pair = lane + part * 32;
        half2 q = __float2half2_rn(0.0f);
        if (owned)
        {
            const auto* source = p.query + (std::size_t(row) * p.query_heads + head) * 256 + pair * 2;
            q = __floats2half2_rn(source[0] * 0.0625f, source[1] * 0.0625f);
        }
        queries[wave][pair] = q;
    }

    float maximum = -INFINITY;
    float denominator = 0.0f;
    float2 totals_even[4]{};
    float2 totals_odd[4]{};
    int consumed = 0;
    const auto* indices = p.selected + std::size_t(row) * QwenSparseSelected;

    const int t0 = lane;
    const int t1 = lane + 32;

    for (int begin = first, iteration = 0; (Parts == 33) ? (iteration == 0) : (begin < total);
         begin += Parts * 64, ++iteration)
    {
        const int count = min(64, total - begin);
        if (t0 < count)
        {
            const int logical = indices[begin + t0];
            if (allVisible)
                require(logical == begin + t0);
            else
                require(logical >= 0 && logical < visible &&
                        (begin + t0 == 0 || logical > indices[begin + t0 - 1]));
        }
        if (t1 < count)
        {
            const int logical = indices[begin + t1];
            if (allVisible)
                require(logical == begin + t1);
            else
                require(logical >= 0 && logical < visible &&
                        (begin + t1 == 0 || logical > indices[begin + t1 - 1]));
        }

        float score0 = 0.0f;
        float score1 = 0.0f;
        loadSparseHalfKeys<Transform, 0, (Parts == 33), Threads>(
            p, r, indices + begin, count, begin, allVisible, kvHead, origin, capacity, tile);
        __syncthreads();

        if (owned)
        {
            accumulateSparseHalfQKPanel(
                queries[wave], tile + t0 * 68, tile + t1 * 68, score0, score1);
        }
        __syncthreads();

        loadSparseHalfKeys<Transform, 128, (Parts == 33), Threads>(
            p, r, indices + begin, count, begin, allVisible, kvHead, origin, capacity, tile);
        __syncthreads();

        float tileMaximum = 0.0f;
        float tileDenominator = 0.0f;
        if (owned)
        {
            accumulateSparseHalfQKPanel(
                queries[wave] + 64, tile + t0 * 68, tile + t1 * 68, score0, score1);
            if (t0 >= count) score0 = -INFINITY;
            if (t1 >= count) score1 = -INFINITY;
            tileMaximum = fmaxf(score0, score1);
#pragma unroll
            for (int distance = 16; distance > 0; distance >>= 1)
                tileMaximum = fmaxf(tileMaximum, __shfl_xor_sync(0xffffffffu, tileMaximum, distance, 32));
            const float exp0 = (t0 < count) ? expf(score0 - tileMaximum) : 0.0f;
            const float exp1 = (t1 < count) ? expf(score1 - tileMaximum) : 0.0f;
            probabilities[wave][t0] = __float2half_rn(exp0 * 0x1p-7f);
            probabilities[wave][t1] = __float2half_rn(exp1 * 0x1p-7f);
            tileDenominator = exp0 + exp1;
#pragma unroll
            for (int distance = 16; distance > 0; distance >>= 1)
                tileDenominator += __shfl_xor_sync(0xffffffffu, tileDenominator, distance, 32);
            tileDenominator *= 0x1p-7f;
        }
        __syncthreads();

        half2 weighted_even[4];
        half2 weighted_odd[4];
        if (owned)
        {
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                weighted_even[c] = __float2half2_rn(0.0f);
                weighted_odd[c] = __float2half2_rn(0.0f);
            }
        }
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk)
        {
            const int chunk_offset = min(chunk * 32, count);
            const int chunk_count = max(0, min(32, count - chunk * 32));
            loadSparseHalfValues<(Parts == 33), Threads>(
                p, indices + begin + chunk_offset, chunk_count,
                begin + chunk_offset, allVisible, kvHead, origin, capacity, tile);
            __syncthreads();
            if (owned)
            {
                if constexpr (Parts == 33)
                {
                    if (count == 64)
                        accumulateSparseHalfValuePanel<true>(
                            probabilities[wave], tile, chunk, lane, count, weighted_even, weighted_odd);
                    else
                        accumulateSparseHalfValuePanel<false>(
                            probabilities[wave], tile, chunk, lane, count, weighted_even, weighted_odd);
                }
                else
                {
                    accumulateSparseHalfValuePanel<false>(
                        probabilities[wave], tile, chunk, lane, count, weighted_even, weighted_odd);
                }
            }
            if (chunk == 0 || begin + Parts * 64 < total)
                __syncthreads();
        }
        if (owned)
        {
            const float nextMaximum = fmaxf(maximum, tileMaximum);
            const auto scales = sparseHalfOnlineScales(maximum, tileMaximum, denominator, nextMaximum);
            const float previousScale = scales.previous;
            const float tileScale = scales.tile;
#pragma unroll
            for (int c = 0; c < 4; ++c)
            {
                const float2 ve = __half22float2(weighted_even[c]);
                const float2 vo = __half22float2(weighted_odd[c]);
                totals_even[c].x = fmaf(totals_even[c].x, previousScale, ve.x * tileScale);
                totals_even[c].y = fmaf(totals_even[c].y, previousScale, ve.y * tileScale);
                totals_odd[c].x = fmaf(totals_odd[c].x, previousScale, vo.x * tileScale);
                totals_odd[c].y = fmaf(totals_odd[c].y, previousScale, vo.y * tileScale);
            }
            denominator = fmaf(denominator, previousScale, tileDenominator * tileScale);
            maximum = nextMaximum;
            consumed += count;
        }
    }

    if (owned)
    {
        auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * Parts + blockIdx.z];
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            partial.weighted[(lane * 4 + c) * 2] = totals_even[c].x + totals_odd[c].x;
            partial.weighted[(lane * 4 + c) * 2 + 1] = totals_even[c].y + totals_odd[c].y;
        }
        if (lane == 0)
        {
            partial.maximum = maximum;
            partial.denominator = denominator;
            const bool firstConsumer = group + wave == max(group, p.query_head_start);
            partial.key_elements = firstConsumer ? consumed * 256 : 0;
            partial.value_elements = firstConsumer ? consumed * 256 : 0;
        }
    }
}
} // namespace llaminar2::qwen4exp::cuda_sparse_detail
