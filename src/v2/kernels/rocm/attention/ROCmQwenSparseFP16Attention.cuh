/**
 * @file ROCmQwenSparseFP16Attention.cuh
 * @brief Native FP16 sparse attention with fixed phase-specific shared KV tiles.
 *
 * The 8-part program shares each KV tile across four fixed heads; phase33 uses six. One wave owns a
 * head and one lane owns a complete QK score, using ordered packed-half dots
 * with FP32 accumulation. The 8-part program retains eight immutable interleaved tile
 * streams; the explicit phase33 program assigns one 64-token tile per part.
 * Each program has fixed arithmetic for every admitted M and compact TP shard. Tile-local half
 * PV chains widen into FP32 online totals, so long contexts cannot overflow a
 * persistent half accumulator. All storage is caller-owned or static LDS.
 *
 * The tile organization follows the MIT-licensed llama.cpp/Castagna tile
 * design (ggml-cuda/fattn-tile.cuh, Castagna revision3483d462). Native selected
 * addressing, pre-RoPE reads, fixed partitions and owned-head publication are
 * Llaminar contracts; no dense gather or host-owned execution state is added.
 */
#pragma once
#include "ROCmQwenSparseOnlineRescale.cuh"

namespace llaminar2::qwen4exp::rocm_sparse_detail
{
/** @brief Four naturally aligned packed-half pairs for one vector tile transfer. */
struct alignas(16) SparseHalfVector
{
    half2 pairs[4];
};
static_assert(sizeof(SparseHalfVector) == 16);

/**
 * @brief Thread-private raw KV panel reused across the QK and PV overlap windows.
 * @details G6 uses 384 cooperative threads for 1024 vectors. The first 256
 * threads own three vectors and the remaining 128 own two. Unused slots are
 * initialized but never published. K1, V0 and V1 each die at their LDS
 * publication before the same bank is reused by the next panel. This is intended to scalarize into registers;
 * compiler spill/resource and ISA gates must prove that physical realization.
 * It adds no workspace, allocation or native memory transfer.
 */
struct SparseHalfPanelPrefetch
{
    SparseHalfVector first, second, third;
};
static_assert(sizeof(SparseHalfPanelPrefetch) == 48);

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
 * @brief One guarded native KV-vector source, prepared before the value is loaded.
 * @details An inactive source owns no native read. Only the key/value preparation helpers
 * create live sources, after the original count, selected-ID and ring logic.
 * This thread-private descriptor is neither model state nor workspace.
 */
struct SparseHalfVectorSource
{
    const SparseHalfVector* data = nullptr;
    bool live = false;
};

/**
 * @brief Read one prepared native vector or construct the original neutral bits.
 * @param source Thread-private source from the original count/address proof.
 * @return The exact raw16-byte value or four original positive-zero half pairs.
 * @details This shared K1/V payload-load authority serves their ordinary copies and
 * fixed prefetch field. It adds no arithmetic, conversion or padded native read.
 */
__device__ __forceinline__ SparseHalfVector readSparseHalfVector(
    const SparseHalfVectorSource& source)
{
    SparseHalfVector value;
#pragma unroll
    for (auto& pairValue : value.pairs) pairValue = __float2half2_rn(0.0f);
    if (source.live) value = *source.data;
    return value;
}

/**
 * @brief Original selected logical ID and its derived physical key row.
 * @details Inactive records have no selected/native read. Logical IDs retain
 * the original pre-RoPE position while K0 runs; only physical rows are consumed
 * by K1 after K0 publication, so logical values have no later live consumer.
 */
struct SparseHalfKeyRow
{
    int logical = -1;
    int physical = -1;
};

/** @brief Typed direct row source for the unchanged interleaved8 schedule. */
struct SparseHalfDirectKeyRows
{
    static constexpr bool retainsRows = false;
};

/**
 * @brief Three thread-owned derived key rows for one immutable33 logical tile.
 * @details This is ephemeral register state for a single invocation, not a
 * request/model authority or workspace. The third row is inactive for the last
 * 128 G6 threads. K1 consumes only physical fields before its payloads are born.
 */
struct SparseHalfRetainedKeyRows
{
    static constexpr bool retainsRows = true;
    SparseHalfKeyRow first, second, third;
};

/**
 * @brief Resolve the original selected ID and ring row once for a key token.
 * @tparam FullPanel Original full64 permission.
 * @tparam IdentityAddressing Original validated identity membership.
 * @param selected Original selected list at this logical tile's first token.
 * @param count Original live key count.
 * @param logicalBegin Original selected ordinal at this tile's first token.
 * @param origin Original ring origin.
 * @param capacity Original ring capacity, or zero for contiguous keys.
 * @param token Original key token in [0,64).
 * @return Original logical/physical pair, or an inactive record with no read.
 * @details This is the single key selected/ring authority for both direct and
 * retained-row copies. No inverse-ring reconstruction or ordinal substitution
 * is used for sparse lists or pre-RoPE positions.
 */
template<bool FullPanel, bool IdentityAddressing>
__device__ __forceinline__ SparseHalfKeyRow prepareSparseHalfKeyRow(
    const std::int32_t* selected, int count, int logicalBegin,
    int origin, int capacity, int token)
{
    SparseHalfKeyRow row{};
    if (FullPanel || token < count)
    {
        row.logical = IdentityAddressing ? logicalBegin + token : selected[token];
        row.physical = detail::physicalRow(row.logical, origin, capacity);
    }
    return row;
}

/**
 * @brief Evaluate the original direct key-row lookup at the original vector.
 * @tparam FullPanel Original full64 permission.
 * @tparam IdentityAddressing Original membership permission.
 * @param rows Empty typed source; it retains no invocation values.
 * @param selected Original selected list.
 * @param count Original live key count.
 * @param logicalBegin Original selected ordinal at the tile start.
 * @param origin Original ring origin.
 * @param capacity Original ring capacity or zero.
 * @param vector Original admitted64-by16 key vector.
 * @param slot Unused for direct lookup; optimized away with the empty source.
 * @return Original logical/physical row for this vector's token.
 */
template<bool FullPanel, bool IdentityAddressing>
__device__ __forceinline__ SparseHalfKeyRow sparseHalfKeyRowForVector(
    [[maybe_unused]] const SparseHalfDirectKeyRows& rows, const std::int32_t* selected,
    int count, int logicalBegin, int origin, int capacity, int vector, [[maybe_unused]] int slot)
{
    return prepareSparseHalfKeyRow<FullPanel, IdentityAddressing>(
        selected,count,logicalBegin,origin,capacity,vector / 16);
}

/**
 * @brief Borrow an already resolved thread-owned row without a selected read.
 * @tparam FullPanel Original permission, already honored during preparation.
 * @tparam IdentityAddressing Original permission, already honored during preparation.
 * @param rows Original three row records prepared before K0.
 * @param selected Unused original selected binding; no native/selected read occurs.
 * @param count Original count, already guarded by the caller and preparation.
 * @param logicalBegin Original ordinal, already included in the retained records.
 * @param origin Original ring origin, already resolved by the same authority.
 * @param capacity Original ring capacity, already resolved by the same authority.
 * @param vector Original cooperative vector; its slot has unchanged ownership.
 * @param slot Compile-unrolled original vector ordinal0,1,2 within this thread.
 * @return The original retained logical/physical row for that slot.
 */
template<bool FullPanel, bool IdentityAddressing>
__device__ __forceinline__ SparseHalfKeyRow sparseHalfKeyRowForVector(
    const SparseHalfRetainedKeyRows& rows, [[maybe_unused]] const std::int32_t* selected,
    [[maybe_unused]] int count, [[maybe_unused]] int logicalBegin, [[maybe_unused]] int origin,
    [[maybe_unused]] int capacity, [[maybe_unused]] int vector, int slot)
{
    if (slot == 0) return rows.first;
    if (slot == 1) return rows.second;
    return rows.third;
}

/**
 * @brief Resolve the three original G6 row slots before any K0 payload read.
 * @tparam FullPanel Original full64 permission.
 * @tparam IdentityAddressing Original membership permission.
 * @tparam Threads Exact384-thread G6 ownership.
 * @param selected Original selected list at this tile's first token.
 * @param count Original live count.
 * @param logicalBegin Original tile start ordinal.
 * @param origin Original ring origin.
 * @param capacity Original ring capacity or zero.
 * @return Three named rows, with an inactive unused third field where required.
 */
template<bool FullPanel, bool IdentityAddressing, int Threads>
__device__ __forceinline__ SparseHalfRetainedKeyRows prepareSparseHalfKeyRowsBody(
    const std::int32_t* selected, int count, int logicalBegin, int origin, int capacity)
{
    static_assert(Threads==384,"Retained key rows require G6 ownership");
    const int vector = int(threadIdx.x);
    SparseHalfRetainedKeyRows rows{};
    rows.first = prepareSparseHalfKeyRow<FullPanel, IdentityAddressing>(
        selected,count,logicalBegin,origin,capacity,vector / 16);
    rows.second = prepareSparseHalfKeyRow<FullPanel, IdentityAddressing>(
        selected,count,logicalBegin,origin,capacity,(vector + Threads) / 16);
    if (vector + 2 * Threads < 64 * 16)
        rows.third = prepareSparseHalfKeyRow<FullPanel, IdentityAddressing>(
            selected,count,logicalBegin,origin,capacity,(vector + 2 * Threads) / 16);
    return rows;
}

/**
 * @brief Seal original identity/full permissions for the retained key rows.
 * @tparam Threads Exact G6 cooperative size.
 * @param selected Original selected list at this logical tile.
 * @param count Original live key count.
 * @param logicalBegin Original selected start ordinal.
 * @param allVisible Original validated identity proof.
 * @param origin Original ring origin.
 * @param capacity Original ring capacity or zero.
 * @return Register candidates for the original physical rows and K0 positions.
 */
template<int Threads>
__device__ __forceinline__ SparseHalfRetainedKeyRows prepareSparseHalfKeyRows(
    const std::int32_t* selected, int count, int logicalBegin, bool allVisible,
    int origin, int capacity)
{
    if (allVisible)
    {
        if (count == 64)
            return prepareSparseHalfKeyRowsBody<true,true,Threads>(selected,count,logicalBegin,origin,capacity);
        return prepareSparseHalfKeyRowsBody<false,true,Threads>(selected,count,logicalBegin,origin,capacity);
    }
    if (count == 64)
        return prepareSparseHalfKeyRowsBody<true,false,Threads>(selected,count,logicalBegin,origin,capacity);
    return prepareSparseHalfKeyRowsBody<false,false,Threads>(selected,count,logicalBegin,origin,capacity);
}

/**
 * @brief Resolve one raw second-key-panel vector without reading its payload.
 * @tparam FullPanel True only when all64 selected key tokens are live.
 * @tparam IdentityAddressing Use the original validated identity membership.
 * @tparam KeyRows Typed direct or original retained key-row source.
 * @param p Original immutable native FP16 key bank.
 * @param selected Sorted selected IDs at this logical tile's first token.
 * @param count Original live key tokens, from one through64.
 * @param logicalBegin Original selected ordinal at the tile's first token.
 * @param kvHead Original local KV owner.
 * @param origin Original physical ring origin.
 * @param capacity Original ring capacity, or zero for contiguous keys.
 * @param vector Admitted vector in [0,1024), mapped as64 tokens by16 vectors.
 * @param rows Typed direct source or original thread-owned rows; no host mirror.
 * @param slot Original unrolled vector ordinal within this thread.
 * @return A guarded native source for coordinates128..255, with no payload read.
 * @details This is the single raw K1 address authority for both copy schedules.
 * The existing transform applies only to Coordinate0; Coordinate128 remains
 * raw in both admitted key encodings. No value-bank binding is substituted.
 */
template<bool FullPanel, bool IdentityAddressing, typename KeyRows=SparseHalfDirectKeyRows>
__device__ __forceinline__ SparseHalfVectorSource prepareSparseHalfKeySecondPanelVector(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, int kvHead, int origin, int capacity, int vector, int slot, const KeyRows& rows = {})
{
    const int token = vector / 16, pair = (vector % 16) * 4;
    SparseHalfVectorSource source{};
    if (FullPanel || token < count)
    {
        const auto row = sparseHalfKeyRowForVector<FullPanel, IdentityAddressing>(
            rows,selected,count,logicalBegin,origin,capacity,vector,slot);
        const int physical = row.physical;
        const auto* bytes = static_cast<const unsigned char*>(p.kv.keys.data) +
            std::size_t(physical) * p.kv.keys.token_stride_bytes +
            std::size_t(kvHead) * p.kv.keys.head_stride_bytes +
            std::size_t(128 + pair * 2) * sizeof(half);
        source.data = reinterpret_cast<const SparseHalfVector*>(bytes);
        source.live = true;
    }
    return source;
}

/**
 * @brief Load one selected 64-token, 128-coordinate key panel into padded LDS.
 * @tparam Transform Whether the admitted native key bank is stored before RoPE.
 * @tparam Coordinate First coordinate of this immutable 128-coordinate panel.
 * @tparam LoadUnroll Vector-transfer scheduling factor: original1 or phase33 factor4.
 * @tparam FullPanel True only when the wrapper proves all64 key tokens live.
 * @tparam Threads Immutable cooperative copy stride: 256 for G4 or 384 for G6.
 * @tparam IdentityAddressing Use validated original ordinals instead of selected IDs.
 * @tparam KeyRows Typed direct or original retained key-row source.
 * @param p Original immutable attention bindings and native cache policy.
 * @param r Authenticated original replay, including absolute position origin.
 * @param selected Sorted selected indices at this tile's first logical token.
 * @param count Live tile tokens, from one through64; padding is never read.
 * @param logicalBegin Original selected ordinal of this panel's first token.
 * @param kvHead Owned local native KV head shared by this CTA's query heads.
 * @param origin Canonical physical ring origin for the current query row.
 * @param capacity Canonical physical ring capacity, or zero for contiguous KV.
 * @param tile Exclusive panel storage with68 packed pairs per token.
 * @param rows Typed direct source or original thread-owned rows; no host mirror.
 * @details Each selected native element is read once per group. The original
 * readKeyAs rounding boundaries are retained before FP16 tile publication.
 * The caller publishes the entire panel with a CTA barrier before consuming it.
 * All-visible addressing uses the independently validated original ordinal.
 * Sparse lists retain selected-ID loads; physical ring mapping and absolute
 * RoPE positions are unchanged. Production8 always passes constant false.
 */
template<bool Transform, int Coordinate, int LoadUnroll, bool FullPanel, int Threads, bool IdentityAddressing=false, typename KeyRows=SparseHalfDirectKeyRows>
__device__ __forceinline__ void copySparseHalfKeysPanel(
    const QwenSparseAttention& p, const QwenSparseReplay& r,
    const std::int32_t* selected, int count, int logicalBegin,
    int kvHead, int origin, int capacity, half2* tile, const KeyRows& rows = {})
{
    static_assert(LoadUnroll==1 || LoadUnroll==4,"Closed native tile-load schedule");
    static_assert(Threads==256 || Threads==384,"Closed cooperative tile-copy geometry");
    static_assert(!KeyRows::retainsRows || Threads==384,"Retained key rows require G6 ownership");
#pragma unroll(LoadUnroll)
    for (int vector = int(threadIdx.x), slot = 0; vector < 64 * 16; vector += Threads, ++slot)
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
                if (FullPanel || token < count)
                {
                    const auto row = sparseHalfKeyRowForVector<FullPanel, IdentityAddressing>(
                        rows,selected,count,logicalBegin,origin,capacity,vector,slot);
                    const int logical = row.logical;
                    const int physical = row.physical;
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
        if constexpr (Coordinate == 128)
        {
            const auto source = prepareSparseHalfKeySecondPanelVector<FullPanel, IdentityAddressing>(
                p,selected,count,logicalBegin,kvHead,origin,capacity,vector,slot,rows);
            value = readSparseHalfVector(source);
        }
        else
        {
#pragma unroll
            for (auto& pairValue : value.pairs) pairValue = __float2half2_rn(0.0f);
            if (FullPanel || token < count)
            {
                const auto row = sparseHalfKeyRowForVector<FullPanel, IdentityAddressing>(
                    rows,selected,count,logicalBegin,origin,capacity,vector,slot);
                const int physical = row.physical;
                const auto* bytes = static_cast<const unsigned char*>(p.kv.keys.data) +
                    std::size_t(physical) * p.kv.keys.token_stride_bytes +
                    std::size_t(kvHead) * p.kv.keys.head_stride_bytes +
                    std::size_t(Coordinate + pair * 2) * sizeof(half);
                value = *reinterpret_cast<const SparseHalfVector*>(bytes);
            }
        }
        *reinterpret_cast<SparseHalfVector*>(tile + token * 68 + pair) = value;
    }
}

/**
 * @brief Dispatch one shared key-copy body with explicit full-panel permission.
 * @tparam Transform Whether the admitted native key bank is stored before RoPE.
 * @tparam Coordinate First coordinate of this immutable 128-coordinate panel.
 * @tparam LoadUnroll Existing vector scheduling factor, independent of permission.
 * @tparam AllowFullPanel True only for the explicit Parts33 producer.
 * @tparam Threads Exact CTA size shared by copy wrappers and their panel body.
 * @tparam KeyRows Typed direct or original retained key-row source.
 * @param p Original immutable attention bindings and native cache policy.
 * @param r Authenticated original replay, including absolute position origin.
 * @param selected Sorted selected indices at this tile's first logical token.
 * @param count Live tile tokens, from one through64; padding is never read.
 * @param logicalBegin Original selected ordinal of this panel's first token.
 * @param allVisible Row-uniform identity proof, enabled only for the explicit phase33 program.
 * @param kvHead Owned local native KV head shared by this CTA's query heads.
 * @param origin Canonical physical ring origin for the current query row.
 * @param capacity Canonical physical ring capacity, or zero for contiguous KV.
 * @param tile Exclusive panel storage with68 packed pairs per token.
 * @param rows Typed direct source or original thread-owned rows; no host mirror.
 * @details Each selected native element is read once per group. The original
 * readKeyAs rounding boundaries are retained before FP16 tile publication.
 * The caller publishes the entire panel with a CTA barrier before consuming it.
 * Full64 panels specialize the body only with explicit permission; otherwise
 * every original count guard remains. Dispatch adds no synchronization.
 * All-visible addressing uses the independently validated original ordinal.
 * Sparse lists retain selected-ID loads; physical ring mapping and absolute
 * RoPE positions are unchanged. Production8 always passes constant false.
 */
template<bool Transform, int Coordinate, int LoadUnroll=1, bool AllowFullPanel=false, int Threads=256, typename KeyRows=SparseHalfDirectKeyRows>
__device__ __forceinline__ void loadSparseHalfKeys(
    const QwenSparseAttention& p, const QwenSparseReplay& r,
    const std::int32_t* selected, int count, int logicalBegin, bool allVisible,
    int kvHead, int origin, int capacity, half2* tile, const KeyRows& rows = {})
{
    if constexpr (AllowFullPanel)
    {
        // Resolve the row-uniform address proof once per panel, outside every
        // unrolled vector transfer. Both routes retain the same copy body.
        if (allVisible)
        {
            if (count == 64)
                copySparseHalfKeysPanel<Transform, Coordinate, LoadUnroll, true, Threads, true>(p,r,selected,count,logicalBegin,kvHead,origin,capacity,tile,rows);
            else
                copySparseHalfKeysPanel<Transform, Coordinate, LoadUnroll, false, Threads, true>(p,r,selected,count,logicalBegin,kvHead,origin,capacity,tile,rows);
            return;
        }
        if (count == 64)
        {
            copySparseHalfKeysPanel<Transform, Coordinate, LoadUnroll, true, Threads>(p,r,selected,count,logicalBegin,kvHead,origin,capacity,tile,rows);
            return;
        }
    }
    copySparseHalfKeysPanel<Transform, Coordinate, LoadUnroll, false, Threads>(p,r,selected,count,logicalBegin,kvHead,origin,capacity,tile,rows);
}

/**
 * @brief Prepare all guarded K1 addresses, then read three fixed register fields.
 * @tparam FullPanel Original full64 permission; partial tiles retain every guard.
 * @tparam IdentityAddressing Original row-uniform identity membership proof.
 * @tparam Threads Exact384-thread G6 cooperative ownership.
 * @tparam KeyRows Typed direct or original retained key-row source.
 * @param p Original immutable attention bindings.
 * @param selected Original sorted IDs at the logical tile's first token.
 * @param count Original live tile count.
 * @param logicalBegin Original selected ordinal at the tile's first token.
 * @param kvHead Original native key head.
 * @param origin Original ring origin.
 * @param capacity Original ring capacity, or zero for contiguous keys.
 * @param rows Typed direct source or original thread-owned rows; no host mirror.
 * @return One48-byte bank; last128 threads have a neutral unused third field.
 * @details No LDS is written. The caller keeps both original publication and
 * retirement barriers; actual ISA must prove native loads overlap QK0.
 */
template<bool FullPanel, bool IdentityAddressing, int Threads, typename KeyRows=SparseHalfDirectKeyRows>
__device__ __forceinline__ SparseHalfPanelPrefetch prefetchSparseHalfKeySecondPanelBody(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, int kvHead, int origin, int capacity, const KeyRows& rows = {})
{
    static_assert(Threads==384,"K1 prefetch requires G6 ownership");
    const int vector = int(threadIdx.x);
    const auto first = prepareSparseHalfKeySecondPanelVector<FullPanel, IdentityAddressing>(
        p,selected,count,logicalBegin,kvHead,origin,capacity,vector,0,rows);
    const auto second = prepareSparseHalfKeySecondPanelVector<FullPanel, IdentityAddressing>(
        p,selected,count,logicalBegin,kvHead,origin,capacity,vector + Threads,1,rows);
    SparseHalfVectorSource third{};
    if (vector + 2 * Threads < 64 * 16)
        third = prepareSparseHalfKeySecondPanelVector<FullPanel, IdentityAddressing>(
            p,selected,count,logicalBegin,kvHead,origin,capacity,vector + 2 * Threads,2,rows);
    return SparseHalfPanelPrefetch{
        readSparseHalfVector(first), readSparseHalfVector(second), readSparseHalfVector(third)};
}

/**
 * @brief Select original full/partial and identity/sparse K1 copy permissions.
 * @tparam Threads Exact G6 cooperative copy size.
 * @tparam KeyRows Typed direct or original retained key-row source.
 * @param p Original immutable attention bindings.
 * @param selected Original sorted IDs at the logical tile's first token.
 * @param count Original live tile count, from one through64.
 * @param logicalBegin Original selected ordinal at the tile's first token.
 * @param allVisible Original validated row-uniform identity proof.
 * @param kvHead Original native key head.
 * @param origin Original ring origin.
 * @param capacity Original ring capacity, or zero for contiguous keys.
 * @param rows Typed direct source or original thread-owned rows; no host mirror.
 * @return One typed K1 bank, without publishing or changing membership.
 */
template<int Threads, typename KeyRows=SparseHalfDirectKeyRows>
__device__ __forceinline__ SparseHalfPanelPrefetch prefetchSparseHalfKeySecondPanel(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, bool allVisible, int kvHead, int origin, int capacity, const KeyRows& rows = {})
{
    if (allVisible)
    {
        if (count == 64)
            return prefetchSparseHalfKeySecondPanelBody<true, true, Threads>(p,selected,count,logicalBegin,kvHead,origin,capacity,rows);
        return prefetchSparseHalfKeySecondPanelBody<false, true, Threads>(p,selected,count,logicalBegin,kvHead,origin,capacity,rows);
    }
    if (count == 64)
        return prefetchSparseHalfKeySecondPanelBody<true, false, Threads>(p,selected,count,logicalBegin,kvHead,origin,capacity,rows);
    return prefetchSparseHalfKeySecondPanelBody<false, false, Threads>(p,selected,count,logicalBegin,kvHead,origin,capacity,rows);
}

/**
 * @brief Publish one admitted K1 vector using its original stride68 LDS layout.
 * @param value Exact raw16-byte register field for coordinates128..255.
 * @param vector Original admitted cooperative vector in [0,1024).
 * @param tile Original key panel after every K0 reader has retired.
 */
__device__ __forceinline__ void publishSparseHalfKeySecondPanelVector(
    const SparseHalfVector& value, int vector, half2* tile)
{
    const int token = vector / 16, pair = (vector % 16) * 4;
    *reinterpret_cast<SparseHalfVector*>(tile + token * 68 + pair) = value;
}

/**
 * @brief Publish the fixed K1 fields after the original K0 retirement barrier.
 * @param prefetched One typed raw panel, reused later for V0 and V1.
 * @param tile Original stride68 key LDS; no V-panel mapping is substituted.
 * @details Every useful/padded vector retains its original owner. The caller
 * retains the original K1 publication barrier before any second-QK reader.
 */
__device__ __forceinline__ void publishSparseHalfKeySecondPanel(
    const SparseHalfPanelPrefetch& prefetched, half2* tile)
{
    const int vector = int(threadIdx.x);
    publishSparseHalfKeySecondPanelVector(prefetched.first, vector, tile);
    publishSparseHalfKeySecondPanelVector(prefetched.second, vector + 384, tile);
    if (vector + 2 * 384 < 64 * 16)
        publishSparseHalfKeySecondPanelVector(prefetched.third, vector + 2 * 384, tile);
}

/**
 * @brief Resolve the original native V-vector address without reading its payload.
 * @tparam FullPanel True only after proving all 32 value tokens live.
 * @tparam IdentityAddressing Use the original validated identity membership.
 * @param p Original immutable native FP16 value binding.
 * @param selected Sorted selected indices at this value panel's first token.
 * @param count Live panel tokens; zero admits no native address or selected read.
 * @param logicalBegin Original selected ordinal of the panel's first token.
 * @param kvHead Original owned native KV head.
 * @param origin Original physical ring origin.
 * @param capacity Original ring capacity, or zero for contiguous KV.
 * @param vector Admitted panel vector in [0,1024); callers own that bound.
 * @return Live native source or a neutral inactive source, with no payload read.
 * @details This is the single selected/ring/stride authority shared by direct
 * copies and prefetch. Preparing all three prefetch sources first gives the
 * compiler a chance to finish selected-ID waits before issuing native loads.
 */
template<bool FullPanel, bool IdentityAddressing>
__device__ __forceinline__ SparseHalfVectorSource prepareSparseHalfValueVector(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, int kvHead, int origin, int capacity, int vector)
{
    const int token = vector / 32, pair = (vector % 32) * 4;
    SparseHalfVectorSource source{};
    if (FullPanel || token < count)
    {
        const int logical = IdentityAddressing ? logicalBegin + token : selected[token];
        const int physical = detail::physicalRow(logical, origin, capacity);
        const auto* bytes = static_cast<const unsigned char*>(p.kv.values.data) +
            std::size_t(physical) * p.kv.values.token_stride_bytes +
            std::size_t(kvHead) * p.kv.values.head_stride_bytes + std::size_t(pair) * sizeof(half2);
        source.data = reinterpret_cast<const SparseHalfVector*>(bytes);
        source.live = true;
    }
    return source;
}

/**
 * @brief Load32 selected value rows into the shared key/value panel.
 * @tparam LoadUnroll Vector-transfer scheduling factor: original1 or phase33 factor4.
 * @tparam FullPanel True only when the wrapper proves all32 value tokens live.
 * @tparam Threads Immutable cooperative copy stride: 256 for G4 or 384 for G6.
 * @tparam IdentityAddressing Use validated original ordinals instead of selected IDs.
 * @tparam Prefetch Return typed V-panel register vectors instead of publishing LDS.
 * @param p Original immutable native FP16 value binding.
 * @param selected Sorted selected indices at this value panel's first token.
 * @param count Live tokens in the panel; zero produces only neutral LDS values.
 * @param logicalBegin Original selected ordinal of this panel's first token.
 * @param kvHead Owned native KV head shared by all CTA consumers.
 * @param origin Canonical physical ring origin.
 * @param capacity Canonical physical ring capacity, or zero for contiguous KV.
 * @param tile Exclusive32-by128 packed-half panel, reused after key retirement.
 * @return A thread-private register bank for Prefetch, otherwise void.
 * @details Vector loads read only live selected values. The caller owns the
 * publication and retirement barriers, so this helper never synchronizes.
 * Identity rows use validated original ordinals; sparse rows read selected IDs.
 * Count-zero panels read neither route. Both preserve the original ring map.
 */
template<int LoadUnroll, bool FullPanel, int Threads, bool IdentityAddressing=false, bool Prefetch=false>
__device__ __forceinline__ auto copySparseHalfValuesPanel(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, int kvHead, int origin, int capacity, half2* tile)
{
    static_assert(LoadUnroll==1 || LoadUnroll==4,"Closed native tile-load schedule");
    static_assert(Threads==256 || Threads==384,"Closed cooperative tile-copy geometry");
    static_assert(!Prefetch || Threads==384,"Value register prefetch requires G6 ownership");
    if constexpr (Prefetch)
    {
        const int vector = int(threadIdx.x);
        const auto first = prepareSparseHalfValueVector<FullPanel, IdentityAddressing>(
            p,selected,count,logicalBegin,kvHead,origin,capacity,vector);
        const auto second = prepareSparseHalfValueVector<FullPanel, IdentityAddressing>(
            p,selected,count,logicalBegin,kvHead,origin,capacity,vector + Threads);
        SparseHalfVectorSource third{};
        if (vector + 2 * Threads < 32 * 32)
            third = prepareSparseHalfValueVector<FullPanel, IdentityAddressing>(
                p,selected,count,logicalBegin,kvHead,origin,capacity,vector + 2 * Threads);
        // Distinct aggregate fields expose independent SSA results, without a
        // dynamic slot or reused payload temporary. Actual ISA must prove
        // that the compiler keeps their native loads outstanding across QK.
        return SparseHalfPanelPrefetch{
            readSparseHalfVector(first),
            readSparseHalfVector(second),
            readSparseHalfVector(third)};
    }
    else
    {
#pragma unroll(LoadUnroll)
        for (int vector = int(threadIdx.x); vector < 32 * 32; vector += Threads)
        {
            const int token = vector / 32, pair = (vector % 32) * 4;
            const auto source = prepareSparseHalfValueVector<FullPanel, IdentityAddressing>(
                p,selected,count,logicalBegin,kvHead,origin,capacity,vector);
            const auto value = readSparseHalfVector(source);
            *reinterpret_cast<SparseHalfVector*>(tile + token * 128 + pair) = value;
        }
    }
}

/**
 * @brief Dispatch one shared value-copy body with explicit full-panel permission.
 * @tparam LoadUnroll Existing vector scheduling factor, independent of permission.
 * @tparam AllowFullPanel True only for the explicit Parts33 producer.
 * @tparam Threads Exact CTA size shared by copy wrappers and their panel body.
 * @tparam Prefetch Return G6 V-panel register vectors without writing LDS.
 * @param p Original immutable native FP16 value binding.
 * @param selected Sorted selected indices at this value panel's first token.
 * @param count Live tokens in the panel; zero produces only neutral LDS values.
 * @param logicalBegin Original selected ordinal of this panel's first token.
 * @param allVisible Row-uniform identity proof, enabled only for the explicit phase33 program.
 * @param kvHead Owned native KV head shared by all CTA consumers.
 * @param origin Canonical physical ring origin.
 * @param capacity Canonical physical ring capacity, or zero for contiguous KV.
 * @param tile Exclusive32-by128 packed-half panel, reused after key retirement.
 * @return A thread-private register bank for Prefetch, otherwise void.
 * @details Vector loads read only live selected values. The caller owns the
 * publication and retirement barriers, so this helper never synchronizes.
 * Identity rows use validated original ordinals; sparse rows read selected IDs.
 * Count-zero panels read neither route. Both preserve the original ring map.
 * Only permitted full32 panels specialize the shared body. Zero and partial
 * counts always retain the original guard and perform no padded native read.
 */
template<int LoadUnroll=1, bool AllowFullPanel=false, int Threads=256, bool Prefetch=false>
__device__ __forceinline__ auto loadSparseHalfValues(
    const QwenSparseAttention& p, const std::int32_t* selected, int count,
    int logicalBegin, bool allVisible, int kvHead, int origin, int capacity, half2* tile)
{
    static_assert(!Prefetch || (AllowFullPanel && Threads==384),
                  "Only the admitted33 G6 value panel may prefetch");
    if constexpr (AllowFullPanel)
    {
        // Identity and sparse addressing specialize the shared body only after
        // the caller's original membership validation has completed.
        if (allVisible)
        {
            if (count == 32)
                return copySparseHalfValuesPanel<LoadUnroll, true, Threads, true, Prefetch>(p,selected,count,logicalBegin,kvHead,origin,capacity,tile);
            else
                return copySparseHalfValuesPanel<LoadUnroll, false, Threads, true, Prefetch>(p,selected,count,logicalBegin,kvHead,origin,capacity,tile);
        }
        if (count == 32)
        {
            return copySparseHalfValuesPanel<LoadUnroll, true, Threads, false, Prefetch>(p,selected,count,logicalBegin,kvHead,origin,capacity,tile);
        }
    }
    return copySparseHalfValuesPanel<LoadUnroll, false, Threads, false, Prefetch>(p,selected,count,logicalBegin,kvHead,origin,capacity,tile);
}

/**
 * @brief Publish one admitted cooperative value vector into its original LDS slot.
 * @param value Exact raw value retained in a fixed register-bank field.
 * @param vector Original cooperative vector index, proven in [0,1024).
 * @param tile The original shared V panel, after all K readers have retired.
 * @details Publication owns no barrier; the caller retains the original edge.
 */
__device__ __forceinline__ void publishSparseHalfValueVector(
    const SparseHalfVector& value, int vector, half2* tile)
{
    const int token = vector / 32, pair = (vector % 32) * 4;
    *reinterpret_cast<SparseHalfVector*>(tile + token * 128 + pair) = value;
}

/**
 * @brief Publish fixed G6 value fields after prior LDS readers retire.
 * @param prefetched Typed thread-private raw values retained by the corresponding prefetch.
 * @param tile The original shared panel after its prior readers have retired.
 * @details First and second fields cover every thread. Only the first256
 * threads own the third field; the others never load or publish that slot.
 * The original publication barrier remains the consumer edge before PV.
 */
__device__ __forceinline__ void publishSparseHalfValuePrefetch(
    const SparseHalfPanelPrefetch& prefetched, half2* tile)
{
    const int vector = int(threadIdx.x);
    publishSparseHalfValueVector(prefetched.first, vector, tile);
    publishSparseHalfValueVector(prefetched.second, vector + 384, tile);
    if (vector + 2 * 384 < 32 * 32)
        publishSparseHalfValueVector(prefetched.third, vector + 2 * 384, tile);
}

/**
 * @brief Accumulate one parity owner's value panel without changing half-FMA order.
 * @tparam FullTile True only after proving the original selected tile has 64 tokens.
 * @tparam TokenUnroll Physical PV scheduling width: original4 or phase33 width2.
 * @param probabilities Published 64-token probability row of this original head.
 * @param tile Published 32-token native FP16 value panel shared by this CTA.
 * @param chunk Original value-panel ordinal, zero or one within the 64-token tile.
 * @param lane Original 64-lane wave lane; its half-wave parity owns even or odd tokens.
 * @param count Original selected tile count in [0,64]; no padded token is read.
 * @param weighted Original four lane-private packed-half accumulators, borrowed.
 * @details Both instantiations share the same ascending token/coordinate body.
 * The full specialization omits only the impossible token-tail condition;
 * the partial specialization preserves every guard and signed-zero boundary.
 * The caller owns all LDS publication/retirement and this helper never fences.
 */
template<bool FullTile, int TokenUnroll=4>
__device__ __forceinline__ void accumulateSparseHalfValuePanel(
    const half* probabilities,const half2* tile,int chunk,int lane,int count,
    half2 (&weighted)[4])
{
    static_assert(TokenUnroll==2 || TokenUnroll==4,"Closed PV token scheduling width");
#pragma unroll(TokenUnroll)
    for (int token = 0; token < 16; ++token)
    {
        const int selected = 2 * token + lane / 32;
        // Partial tiles retain exactly the original real-token guard. Padded
        // +0 updates would change underflowed negative-zero half accumulators.
        if constexpr (!FullTile)
            if (chunk * 32 + selected >= count) continue;
        const half2 probability = __half2half2(probabilities[chunk * 32 + selected]);
#pragma unroll
        for (int coordinate = 0; coordinate < 4; ++coordinate)
            weighted[coordinate] = __hfma2(probability,
                tile[selected * 128 + (lane % 32) * 4 + coordinate], weighted[coordinate]);
    }
}

/**
 * @brief Admit the guard-free PV body only for an explicitly permitted full tile.
 * @tparam AllowFullTile True only for the explicit Parts33 producer.
 * @param probabilities Original head's published probability row.
 * @param tile Original published native value panel.
 * @param chunk Original panel ordinal, zero or one.
 * @param lane Original 64-lane wave lane and parity owner.
 * @param count Original tile count; only exactly 64 selects the full body.
 * @param weighted Borrowed original four packed-half accumulators.
 * @details This dispatch changes neither accumulator ownership nor operation
 * order. Production 8 always instantiates the guarded body. Partial/zero tiles
 * never enter the full body, and no host state or synchronization is consulted.
 */
template<bool AllowFullTile>
__device__ __forceinline__ void accumulateSparseHalfValues(
    const half* probabilities,const half2* tile,int chunk,int lane,int count,
    half2 (&weighted)[4])
{
    // This permission is true only for Parts33. Scheduling changes the load
    // window, never the ascending token/coordinate half-FMA dependencies.
    constexpr int TokenUnroll = AllowFullTile ? 2 : 4;
    if constexpr (AllowFullTile)
        if (count == 64)
        {
            accumulateSparseHalfValuePanel<true, TokenUnroll>(probabilities,tile,chunk,lane,count,weighted);
            return;
        }
    accumulateSparseHalfValuePanel<false, TokenUnroll>(probabilities,tile,chunk,lane,count,weighted);
}

/**
 * @brief Produce fixed batch-invariant FP16 tile streams over native sparse KV.
 * @tparam Transform Whether selected keys require the declared pre-RoPE transform.
 * @tparam Parts Fixed arithmetic streams: the 8-part or explicit phase33 program.
 * @details The phase33 program uses one 64-token virtual tile per nonempty part;
 * its original first recurrence, parity and publication are not simplified.
 * Its admitted selected extent fits 33 tiles, so each nonempty part executes
 * exactly once. The explicit trip count exposes that invariant to the compiler
 * without changing the literal recurrence or the production 8 stream traversal.
 * @param p Persistent original row, head, cache, selection and partial bindings.
 * @details The grid uses rows, intersected canonical head groups and Parts
 * streams. Groups own four heads for production 8 and six for the explicit phase33 program.
 * A64-token tile belongs to (tile_index moduloParts), independent of row capacity,
 * live M, head-shard width or scheduling. The two half-waves accumulate even
 * and odd PV tokens separately, then merge their FP32 totals in that fixed
 * order. Probability scaling by exactly1/128 bounds every32-term half chain
 * even for maximum finite FP16 values; numerator and denominator share this
 * scale. Inactive rows and unowned heads never publish storage. Empty streams
 * publish only zero-denominator metadata, leaving their payload unauthored.
 */
template<bool Transform,int Parts=QwenSparsePartitions>
__global__ __launch_bounds__(64 * (Parts == 33 ? 6 : 4), 2) void selectedAttentionFP16Tiles(QwenSparseAttention p)
{
    static_assert(Parts==8 || Parts==33,"Closed FP16 partition program");
    // Each wave retains one original head's arithmetic. Canonical six-head
    // groups divide each 12-head KV interval, so the phase33 program shares
    // each native panel across six head slots without crossing a KV owner.
    // Production 8 retains four-head groups and the original unroll1 copy hint.
    constexpr int GroupHeads = Parts == 33 ? 6 : 4;
    constexpr int Threads = 64 * GroupHeads;
    static_assert(12 % GroupHeads == 0, "Attention groups must not cross a KV head");
    constexpr int LoadUnroll=Parts==33?4:1;
    constexpr int QKFirstUnroll = Parts == 33 ? 64 : 16;
    constexpr int QKSecondUnroll = Parts == 33 ? 32 : 16;
    static_assert(Parts != 33 || QwenSparseSelected <= Parts * 64,
                  "The single-pass 33 schedule requires at most 33 selected tiles");
    const auto r = replay(p.replay, p.row_capacity, p.token_capacity);
    const int row = int(blockIdx.x), lane = int(threadIdx.x) % 64;
    const int wave = int(threadIdx.x) / 64;
    if (row >= r.live_rows) return;
    const int group = (p.query_head_start / GroupHeads + int(blockIdx.y)) * GroupHeads;
    const int head = group + wave - p.query_head_start;
    const bool owned = head >= 0 && head < p.query_heads;
    // A whole wave owns one global head. Arithmetic can skip unowned waves,
    // but all CTA threads retain cooperative KV transfers and CTA barriers.
    // Query and probability rows are wave-private, so no live head consumes
    // an unowned wave's unauthored arithmetic or probability storage.
    const int kvHead = group / 12 - p.kv_head_start;
    const int total = p.selected_count[row];
    require(total > 0 && total <= QwenSparseSelected);
    // Valid sorted unique membership covering every visible token is identity.
    // Every live ordinal is checked before the original key-publication barrier.
    // Production8 folds this proof mode false and retains original sparse reads.
    const int visible = r.prefix_tokens + row + 1;
    const bool allVisible = Parts == 33 && total == visible;
    const int first = int(blockIdx.z) * 64;
    if (first >= total)
    {
        if (owned && lane == 0)
        {
            auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * Parts + blockIdx.z];
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

    __shared__ __align__(16) half2 queries[GroupHeads][128];
    __shared__ __align__(16) half2 tile[64 * 68];
    __shared__ half probabilities[GroupHeads][64];
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
    for (int begin = first, iteration = 0; Parts == 33 ? iteration == 0 : begin < total;
         begin += Parts * 64, ++iteration)
    {
        const int count = min(64, total - begin);
        if (lane < count)
        {
            const int logical = indices[begin + lane];
            if (allVisible)
                require(logical == begin + lane);
            else
                require(logical >= 0 && logical < r.prefix_tokens + row + 1 &&
                    (begin + lane == 0 || logical > indices[begin + lane - 1]));
        }
        float score = 0.0f;
        SparseHalfRetainedKeyRows retainedKeyRows;
        if constexpr (Parts == 33)
        {
            // Derive rows only after original membership validation. Logical
            // positions serve K0 RoPE; only physical rows survive its publication.
            retainedKeyRows = prepareSparseHalfKeyRows<Threads>(
                indices + begin,count,begin,allVisible,origin,capacity);
            loadSparseHalfKeys<Transform, 0, LoadUnroll, true, Threads>(
                p,r,indices + begin,count,begin,allVisible,kvHead,origin,capacity,tile,retainedKeyRows);
        }
        else
            loadSparseHalfKeys<Transform, 0, LoadUnroll, (Parts == 33), Threads>(p, r, indices + begin, count, begin, allVisible, kvHead, origin, capacity, tile);
        __syncthreads();
        SparseHalfPanelPrefetch prefetchedPanel;
        if constexpr (Parts == 33)
            // K0 and query publication also completes the original validation.
            // Retain raw K1 in registers while QK0 reads only the live LDS panel.
            prefetchedPanel = prefetchSparseHalfKeySecondPanel<Threads>(
                p, indices + begin, count, begin, allVisible, kvHead, origin, capacity, retainedKeyRows);
        using NativePair = _Float16 __attribute__((ext_vector_type(2)));
        if (owned)
        {
            // A wider scheduling window exposes the next aligned LDS reads
            // before the dependent dot chain completes. Dot coordinates and
            // their FP32 accumulation order remain unchanged. The scheduling
            // width is sealed by Parts; production8 retains its original16.
#pragma unroll(QKFirstUnroll)
            for (int pair = 0; pair < 64; ++pair)
                score = __builtin_amdgcn_fdot2(__builtin_bit_cast(NativePair, queries[wave][pair]),
                    __builtin_bit_cast(NativePair, tile[lane * 68 + pair]), score, false);
        }
        __syncthreads(); // Every QK reader retires before the second panel replaces K.
        if constexpr (Parts == 33)
            publishSparseHalfKeySecondPanel(prefetchedPanel, tile);
        else
            loadSparseHalfKeys<Transform, 128, LoadUnroll, (Parts == 33), Threads>(p, r, indices + begin, count, begin, allVisible, kvHead, origin, capacity, tile);
        __syncthreads();
        if constexpr (Parts == 33)
            // All original membership/ring validation and both K publications
            // have completed. Read V0 now without touching the live K LDS bank;
            // its registers span the unchanged second QK/softmax arithmetic.
            prefetchedPanel = loadSparseHalfValues<LoadUnroll, true, Threads, true>(
                p, indices + begin, min(32, count), begin,
                allVisible, kvHead, origin, capacity, tile);
        float tileMaximum = 0.0f, tileDenominator = 0.0f;
        if (owned)
        {
#pragma unroll(QKSecondUnroll)
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
            if constexpr (Parts == 33)
                publishSparseHalfValuePrefetch(prefetchedPanel, tile);
            if constexpr (Parts != 33)
                loadSparseHalfValues<LoadUnroll, (Parts == 33), Threads>(p, indices + begin + min(chunk * 32, count),
                max(0, min(32, count - chunk * 32)), begin + min(chunk * 32, count),
                allVisible, kvHead, origin, capacity, tile);
            __syncthreads();
            if constexpr (Parts == 33)
                if (chunk == 0)
                    // V0 fields died at publication above. Reuse that bank for
                    // V1 while unchanged PV0 consumes only the published LDS.
                    // V1 reaches LDS only after the original V0 retirement edge.
                    prefetchedPanel = loadSparseHalfValues<LoadUnroll, true, Threads, true>(
                        p, indices + begin + min(32, count), max(0, min(32, count - 32)),
                        begin + min(32, count), allVisible, kvHead, origin, capacity, tile);
            if (owned)
            {
                accumulateSparseHalfValues<(Parts == 33)>(
                    probabilities[wave],tile,chunk,lane,count,weighted);
            }
            // Retire V readers only when a later writer reuses this panel.
            // This predicate is CTA-uniform and independent of owned heads; the
            // terminal tile's final panel has no later LDS reader or writer.
            if (chunk == 0 || begin + Parts * 64 < total) __syncthreads();
        }
        if (owned)
        {
            const float nextMaximum = fmaxf(maximum, tileMaximum);
            const auto scales = sparseHalfOnlineScales(maximum, tileMaximum, denominator, nextMaximum);
            const float previousScale = scales.previous;
            const float tileScale = scales.tile;
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
                auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * Parts + blockIdx.z];
                partial.weighted[(lane * 4 + coordinate) * 2] = totals[coordinate].x + otherX;
                partial.weighted[(lane * 4 + coordinate) * 2 + 1] = totals[coordinate].y + otherY;
            }
        }
    }
    if (owned && lane == 0)
    {
        auto& partial = p.gpu_partials[(std::size_t(row) * p.query_heads + head) * Parts + blockIdx.z];
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
