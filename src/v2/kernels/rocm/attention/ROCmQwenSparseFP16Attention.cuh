/**
 * @file ROCmQwenSparseFP16Attention.cuh
 * @brief Explicit FP16 sparse-attention experiment with shared native KV tiles.
 *
 * Four fixed global query heads share each selected KV tile. One wave owns a
 * head and one lane owns a complete QK score, using ordered packed-half dots
 * with FP32 accumulation. Eight immutable interleaved tile streams preserve
 * the arithmetic for every admitted M and compact TP shard. Tile-local half
 * PV chains widen into FP32 online totals, so long contexts cannot overflow a
 * persistent half accumulator. All storage is caller-owned or static LDS.
 *
 * The tile organization follows the MIT-licensed llama.cpp/Castagna tile
 * design (ggml-cuda/fattn-tile.cuh, Castagna revision3483d462). Native selected
 * addressing, pre-RoPE reads, fixed partitions and owned-head publication are
 * Llaminar contracts; no dense gather or host-owned execution state is added.
 */
#pragma once

namespace llaminar2::qwen4exp::rocm_sparse_detail
{
/** @brief Four naturally aligned packed-half pairs for one vector tile transfer. */
struct alignas(16) SparseHalfVector
{
    half2 pairs[4];
};
static_assert(sizeof(SparseHalfVector) == 16);

/** @brief Two packed-half pairs; both split-half inputs use exact8-byte transfers. */
struct alignas(8) SparseHalfDuo
{
    half2 pairs[2];
};
static_assert(sizeof(SparseHalfDuo) == 8);

/**
 * @brief Rotate both outputs of one native FP16 split-half pair with shared factors.
 * @param kv Admitted pre-RoPE FP16 key policy with exactly64 rotary coordinates.
 * @param r Original replay owning the absolute position origin.
 * @param logical Selected logical token; caller has validated its causal extent.
 * @param coordinate Original first-half coordinate in0..31, shared by both outputs.
 * @param low Exact decoded native first-half value.
 * @param high Exact decoded native partner32 coordinates later.
 * @return The two separately half-rounded outputs, widened without loss to FP32.
 * @details This is the literal readKey rounding formula for both coordinates:
 * round coefficients, separately round each product, then round each sum. No
 * half FMA replaces those boundaries. Factors are evaluated once perpair using
 * the same canonical source and absolute position; no head or row batching
 * changes the arithmetic. The caller owns publication of both results to LDS.
 */
__device__ __forceinline__ float2 rotateSparseHalfSplitPair(
    const QwenSparseKV& kv, const QwenSparseReplay& r,
    int logical, int coordinate, float low, float high)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    float cosine, sine;
    detail::rotaryFactors(kv.key_rotary_source, kv.key_cosine, kv.key_sine,
        r.position_origin + logical, coordinate, 64, kv.key_policy.rope_theta,
        cosine, sine, nullptr, kv.key_negative_log_theta);
    const auto representation = QwenSparseScalar::FP16;
    const float c = detail::roundRepresentation(cosine, representation);
    const float s = detail::roundRepresentation(sine, representation);
    const float first = detail::roundRepresentation(
        detail::roundRepresentation(low * c, representation) +
        detail::roundRepresentation((-high) * s, representation), representation);
    const float second = detail::roundRepresentation(
        detail::roundRepresentation(high * c, representation) +
        detail::roundRepresentation(low * s, representation), representation);
    return make_float2(first, second);
}

/**
 * @brief Load one selected 64-token, 128-coordinate key panel into padded LDS.
 * @tparam Transform Whether the admitted native key bank is stored before RoPE.
 * @tparam Coordinate First coordinate of this immutable 128-coordinate panel.
 * @param p Original immutable attention bindings and native cache policy.
 * @param r Authenticated original replay, including absolute position origin.
 * @param selected Sorted selected indices at this tile's first logical token.
 * @param count Live tile tokens, from one through64; padding is never read.
 * @param kvHead Owned local native KV head shared by this CTA's query heads.
 * @param origin Canonical physical ring origin for the current query row.
 * @param capacity Canonical physical ring capacity, or zero for contiguous KV.
 * @param tile Exclusive panel storage with68 packed pairs per token.
 * @details Each selected native element is read once per group. The original
 * readKeyAs rounding boundaries are retained before FP16 tile publication.
 * The caller publishes the entire panel with a CTA barrier before consuming it.
 */
template<bool Transform, int Coordinate>
__device__ __forceinline__ void loadSparseHalfKeys(
    const QwenSparseAttention& p, const QwenSparseReplay& r,
    const std::int32_t* selected, int count, int kvHead, int origin, int capacity,
    half2* tile)
{
#pragma unroll 1
    for (int vector = int(threadIdx.x); vector < 64 * 16; vector += 256)
    {
        const int token = vector / 16, columnVector = vector % 16;
        if constexpr (Transform && Coordinate == 0)
        {
            if (columnVector < 8)
            {
                // Eight owners cover32 split-half pairs: each reads four low
                // and four high values once, evaluates four factors, and owns
                // both destinations. No reader can observe an overwritten raw
                // partner, and no extra CTA barrier or raw LDS bank is needed.
                const int coordinate = columnVector * 4;
                SparseHalfDuo low, high;
#pragma unroll
                for (int item = 0; item < 2; ++item)
                {
                    low.pairs[item] = __float2half2_rn(0.0f);
                    high.pairs[item] = __float2half2_rn(0.0f);
                }
                if (token < count)
                {
                    const int logical = selected[token];
                    const int physical = detail::physicalRow(logical, origin, capacity);
                    const auto* bytes = static_cast<const unsigned char*>(p.kv.keys.data) +
                        std::size_t(physical) * p.kv.keys.token_stride_bytes +
                        std::size_t(kvHead) * p.kv.keys.head_stride_bytes;
                    const auto rawLow = *reinterpret_cast<const SparseHalfDuo*>(bytes + coordinate * sizeof(half));
                    const auto rawHigh = *reinterpret_cast<const SparseHalfDuo*>(bytes + (coordinate + 32) * sizeof(half));
#pragma unroll
                    for (int item = 0; item < 2; ++item)
                    {
                        const auto x = __half22float2(rawLow.pairs[item]);
                        const auto y = __half22float2(rawHigh.pairs[item]);
                        const auto first = rotateSparseHalfSplitPair(p.kv, r, logical, coordinate + item * 2, x.x, y.x);
                        const auto second = rotateSparseHalfSplitPair(p.kv, r, logical, coordinate + item * 2 + 1, x.y, y.y);
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
        for (auto& pairValue : value.pairs) pairValue = __float2half2_rn(0.0f);
        if (token < count)
        {
            const int physical = detail::physicalRow(selected[token], origin, capacity);
            const auto* bytes = static_cast<const unsigned char*>(p.kv.keys.data) +
                std::size_t(physical) * p.kv.keys.token_stride_bytes +
                std::size_t(kvHead) * p.kv.keys.head_stride_bytes +
                std::size_t(Coordinate + pair * 2) * sizeof(half);
            value = *reinterpret_cast<const SparseHalfVector*>(bytes);
        }
        *reinterpret_cast<SparseHalfVector*>(tile + token * 68 + pair) = value;
    }
}

/**
 * @brief Load32 selected value rows into the shared key/value panel.
 * @param p Original immutable native FP16 value binding.
 * @param selected Sorted selected indices at this value panel's first token.
 * @param count Live tokens in the panel; zero produces only neutral LDS values.
 * @param kvHead Owned native KV head shared by all CTA consumers.
 * @param origin Canonical physical ring origin.
 * @param capacity Canonical physical ring capacity, or zero for contiguous KV.
 * @param tile Exclusive32-by128 packed-half panel, reused after key retirement.
 * @details Vector loads read only live selected values. The caller owns the
 * publication and retirement barriers, so this helper never synchronizes.
 */
__device__ __forceinline__ void loadSparseHalfValues(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int kvHead, int origin, int capacity, half2* tile)
{
#pragma unroll 1
    for (int vector = int(threadIdx.x); vector < 32 * 32; vector += 256)
    {
        const int token = vector / 32, pair = (vector % 32) * 4;
        SparseHalfVector value;
#pragma unroll
        for (auto& pairValue : value.pairs) pairValue = __float2half2_rn(0.0f);
        if (token < count)
        {
            const int physical = detail::physicalRow(selected[token], origin, capacity);
            const auto* bytes = static_cast<const unsigned char*>(p.kv.values.data) +
                std::size_t(physical) * p.kv.values.token_stride_bytes +
                std::size_t(kvHead) * p.kv.values.head_stride_bytes + std::size_t(pair) * sizeof(half2);
            value = *reinterpret_cast<const SparseHalfVector*>(bytes);
        }
        *reinterpret_cast<SparseHalfVector*>(tile + token * 128 + pair) = value;
    }
}

/**
 * @brief Produce eight batch-invariant FP16 tile streams over native sparse KV.
 * @tparam Transform Whether selected keys require the declared pre-RoPE transform.
 * @param p Persistent original row, head, cache, selection and partial bindings.
 * @details The grid is [rows,intersected global four-head groups,eight streams].
 * A64-token tile belongs to (tile_index modulo8), independent of row capacity,
 * live M, head-shard width or scheduling. The two half-waves accumulate even
 * and odd PV tokens separately, then merge their FP32 totals in that fixed
 * order. Probability scaling by exactly1/128 bounds every32-term half chain
 * even for maximum finite FP16 values; numerator and denominator share this
 * scale. Inactive rows and unowned heads never publish storage. Empty streams
 * publish only zero-denominator metadata, leaving their payload unauthored.
 */
template<bool Transform>
__global__ __launch_bounds__(256, 2) void selectedAttentionFP16Tiles(QwenSparseAttention p)
{
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    const int row = int(blockIdx.x), lane = int(threadIdx.x) % 64;
    const int wave = int(threadIdx.x) / 64;
    if (row >= r.live_rows) return;
    const int group = (p.query_head_start / 4 + int(blockIdx.y)) * 4;
    const int head = group + wave - p.query_head_start;
    const bool owned = head >= 0 && head < p.query_heads;
    // A whole wave owns one global head. Arithmetic can skip unowned waves,
    // but all 256 threads retain cooperative KV transfers and CTA barriers.
    // Query and probability rows are wave-private, so no live head consumes
    // an unowned wave's unauthored arithmetic or probability storage.
    const int kvHead = group / 12 - p.kv_head_start;
    const int total = p.selected_count[row];
    require(total > 0 && total <= QwenSparseSelected);
    const int first = int(blockIdx.z) * 64;
    if (first >= total)
    {
        if (owned && lane == 0)
        {
            auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * 8 + blockIdx.z];
            partial.maximum = -INFINITY;
            partial.denominator = 0.0f;
            partial.key_elements = partial.value_elements = 0;
        }
        return;
    }
    int origin = 0, capacity = 0;
    require(detail::resolveMapping(p.kv, r, row, p.token_capacity, origin, capacity));
    if constexpr (Transform)
        if (p.kv.key_rotary_source == QwenSparseRotarySource::TablePlanes)
            require(static_cast<long long>(r.position_origin) + r.prefix_tokens + row < p.kv.key_position_capacity);

    __shared__ __align__(16) half2 queries[4][128];
    __shared__ __align__(16) half2 tile[64 * 68];
    __shared__ half probabilities[4][64];
#pragma unroll
    for (int part = 0; part < 2; ++part)
    {
        const int pair = lane + part * 64;
        half2 q = __float2half2_rn(0.0f);
        if (owned)
        {
            const auto* source = p.query + (std::size_t(row) * p.query_heads + head) * 256 + pair * 2;
            q = __floats2half2_rn(source[0] * 0.0625f, source[1] * 0.0625f);
        }
        queries[wave][pair] = q;
    }
    // Query storage is disjoint from the key panel. The first key-publication
    // barrier below also publishes every query pair before any QK reader runs.

    float maximum = -INFINITY, denominator = 0.0f;
    float2 totals[4]{};
    int consumed = 0;
    const auto* indices = p.selected + std::size_t(row) * QwenSparseSelected;
    for (int begin = first; begin < total; begin += 8 * 64)
    {
        const int count = min(64, total - begin);
        if (lane < count)
        {
            const int logical = indices[begin + lane];
            require(logical >= 0 && logical < r.prefix_tokens + row + 1 &&
                (begin + lane == 0 || logical > indices[begin + lane - 1]));
        }
        float score = 0.0f;
        loadSparseHalfKeys<Transform, 0>(p, r, indices + begin, count, kvHead, origin, capacity, tile);
        __syncthreads();
        using NativePair = _Float16 __attribute__((ext_vector_type(2)));
        if (owned)
        {
            // A wider scheduling window exposes the next aligned LDS reads
            // before the dependent dot chain completes. Dot coordinates and
            // their FP32 accumulation order remain unchanged; bounded unrolling
            // limits register lifetimes compared with a fully expanded panel.
#pragma unroll 16
            for (int pair = 0; pair < 64; ++pair)
                score = __builtin_amdgcn_fdot2(__builtin_bit_cast(NativePair, queries[wave][pair]),
                    __builtin_bit_cast(NativePair, tile[lane * 68 + pair]), score, false);
        }
        __syncthreads(); // Every QK reader retires before the second panel replaces K.
        loadSparseHalfKeys<Transform, 128>(p, r, indices + begin, count, kvHead, origin, capacity, tile);
        __syncthreads();
        float tileMaximum = 0.0f, tileDenominator = 0.0f;
        if (owned)
        {
#pragma unroll 16
            for (int pair = 0; pair < 64; ++pair)
                score = __builtin_amdgcn_fdot2(__builtin_bit_cast(NativePair, queries[wave][pair + 64]),
                    __builtin_bit_cast(NativePair, tile[lane * 68 + pair]), score, false);

            if (lane >= count) score = -INFINITY;
            tileMaximum = score;
#pragma unroll
            for (int distance = 32; distance; distance /= 2)
                tileMaximum = fmaxf(tileMaximum, __shfl_xor(tileMaximum, distance, 64));
            const float exponential = lane < count ? expf(score - tileMaximum) : 0.0f;
            probabilities[wave][lane] = __float2half_rn(exponential * 0x1p-7f);
            tileDenominator = exponential;
#pragma unroll
            for (int distance = 32; distance; distance /= 2)
                tileDenominator += __shfl_xor(tileDenominator, distance, 64);
            tileDenominator *= 0x1p-7f;
        }
        __syncthreads(); // Retire K readers and publish every owned probability row.

        half2 weighted[4];
        if (owned)
        {
#pragma unroll
            for (auto& value : weighted) value = __float2half2_rn(0.0f);
        }
#pragma unroll
        for (int chunk = 0; chunk < 2; ++chunk)
        {
            loadSparseHalfValues(p, indices + begin + min(chunk * 32, count),
                max(0, min(32, count - chunk * 32)), kvHead, origin, capacity, tile);
            __syncthreads();
            if (owned)
            {
#pragma unroll 4
                for (int token = 0; token < 16; ++token)
                {
                    const int selected = 2 * token + lane / 32;
                    // The branch is uniform within each logical32-lane PV owner.
                    // Padded +0 updates would alter underflowed negative-zero sums;
                    // only real selected tokens belong to the arithmetic chain.
                    if (chunk * 32 + selected >= count) continue;
                    const half2 probability = __half2half2(probabilities[wave][chunk * 32 + selected]);
#pragma unroll
                    for (int coordinate = 0; coordinate < 4; ++coordinate)
                        weighted[coordinate] = __hfma2(probability,
                            tile[selected * 128 + (lane % 32) * 4 + coordinate], weighted[coordinate]);
                }
            }
            // Retire V readers only when a later writer reuses this panel.
            // This predicate is CTA-uniform and independent of owned heads; the
            // terminal tile's final panel has no later LDS reader or writer.
            if (chunk == 0 || begin + 8 * 64 < total) __syncthreads();
        }
        if (owned)
        {
            const float nextMaximum = fmaxf(maximum, tileMaximum);
            const float previousScale = denominator == 0.0f ? 0.0f : expf(maximum - nextMaximum);
            const float tileScale = expf(tileMaximum - nextMaximum);
#pragma unroll
            for (int coordinate = 0; coordinate < 4; ++coordinate)
            {
                const float2 value = __half22float2(weighted[coordinate]);
                totals[coordinate].x = fmaf(totals[coordinate].x, previousScale, value.x * tileScale);
                totals[coordinate].y = fmaf(totals[coordinate].y, previousScale, value.y * tileScale);
            }
            denominator = fmaf(denominator, previousScale, tileDenominator * tileScale);
            maximum = nextMaximum;
            consumed += count;
        }
    }
    // Ownership is uniform over all 64 lanes: both parity halves of every
    // owned wave execute the unchanged shuffle tree. Unowned waves have no
    // consumer or partial destination and need no final arithmetic.
    if (owned)
    {
#pragma unroll
        for (int coordinate = 0; coordinate < 4; ++coordinate)
        {
            const float otherX = __shfl_xor(totals[coordinate].x, 32, 64);
            const float otherY = __shfl_xor(totals[coordinate].y, 32, 64);
            if (lane < 32)
            {
                auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * 8 + blockIdx.z];
                partial.weighted[(lane * 4 + coordinate) * 2] = totals[coordinate].x + otherX;
                partial.weighted[(lane * 4 + coordinate) * 2 + 1] = totals[coordinate].y + otherY;
            }
        }
    }
    if (owned && lane == 0)
    {
        auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * 8 + blockIdx.z];
        partial.maximum = maximum;
        partial.denominator = denominator;
        // Shared native reads belong to the first actual consumer, once. These
        // are source-level decoded coordinates, not a physical HBM measurement.
        const bool firstConsumer = group + wave == max(group, p.query_head_start);
        // Paired owners load each raw key coordinate once, including both
        // rotary halves. The old scalar reader loaded64 partner coordinates again.
        partial.key_elements = firstConsumer ? consumed * 256 : 0;
        partial.value_elements = firstConsumer ? consumed * 256 : 0;
    }
}
} // namespace llaminar2::qwen4exp::rocm_sparse_detail
