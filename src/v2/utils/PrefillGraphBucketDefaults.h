/**
 * @file PrefillGraphBucketDefaults.h
 * @brief Canonical row buckets and geometry-aware GPU verifier tile policy.
 *
 * This header is deliberately backend independent. Runtime graph planning,
 * NativeVNNI evidence collection, and GPU workspace sizing must agree about
 * the row regimes they support. Keeping those constants and the economical
 * grouped-tile calculation here prevents CUDA and ROCm from silently growing
 * different limits as their kernels are tuned.
 */

#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Canonical graph-prefill bucket sizes used by runtime graph capture
     * and GEMM/GEMV dispatch training.
     */
    inline constexpr size_t kDefaultPrefillGraphBucketCount =
        size_t{0}
#define LLAMINAR_PREFILL_GRAPH_BUCKET(rows) +size_t{1}
#include "utils/PrefillGraphBuckets.def"
#undef LLAMINAR_PREFILL_GRAPH_BUCKET
        ;

    inline constexpr std::array<int, kDefaultPrefillGraphBucketCount>
        kDefaultPrefillGraphBucketSizes = {
#define LLAMINAR_PREFILL_GRAPH_BUCKET(rows) rows,
#include "utils/PrefillGraphBuckets.def"
#undef LLAMINAR_PREFILL_GRAPH_BUCKET
    };

    /**
     * @brief Default number of resident prefill forward-graph identities.
     *
     * The outer forward cache owns complete graph topology and stage-persistent
     * buffers, so this is an admission budget rather than an arbitrary LRU
     * tuning constant. Environment configuration may override it explicitly.
     */
    inline constexpr size_t kDefaultPrefillGraphMaxCachedEntries = 10;

    /**
     * @brief Cache identities reserved beyond the materialized bucket ladder.
     *
     * ExpertOverlay setup materializes every admitted physical row bucket. A
     * live request can additionally bind a request-specific captured identity,
     * while prefix-runtime rehydration may own a second topology. Reserving both
     * keeps ordinary serving from immediately evicting the smallest retained
     * buckets after an otherwise successful setup pass.
     */
    inline constexpr size_t kExpertOverlayPrefillGraphIdentityReserve = 2;

    static_assert(
        kDefaultPrefillGraphMaxCachedEntries >
            kExpertOverlayPrefillGraphIdentityReserve,
        "The default prefill cache must retain at least one overlay bucket");
    static_assert(
        kDefaultPrefillGraphMaxCachedEntries -
                kExpertOverlayPrefillGraphIdentityReserve <=
            kDefaultPrefillGraphBucketCount,
        "The default overlay segment requires enough canonical buckets");

    /**
     * @brief Default maximum physical rows in one ExpertOverlay prefill segment.
     *
     * Select the largest canonical bucket whose complete lower bucket ladder
     * fits beside the reserved runtime identities. This couples the default
     * segment envelope to the cache and bucket sources of truth instead of a
     * duplicated row literal. Users may still tune the limit explicitly.
     */
    inline constexpr int kDefaultExpertOverlayPrefillSegmentRows =
        kDefaultPrefillGraphBucketSizes
            [kDefaultPrefillGraphMaxCachedEntries -
             kExpertOverlayPrefillGraphIdentityReserve - size_t{1}];

    /**
     * @brief Runtime verifier depths certified by the default NativeVNNI sweep.
     *
     * The grouped kernels are runtime-M implementations and do not use this
     * list as a semantic limit. The default training and integration gate sweep
     * every row count through sixteen so a fifteen-draft speculative regime,
     * including its terminal bonus row, is covered without leaving holes
     * between tested depths.
     */
    inline constexpr std::array<int, 15> kDefaultNativeVNNISmallMRows = {
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    /**
     * @brief Default captured verifier row capacity certified in CI.
     *
     * Callers may plan a larger graph/workspace capacity. Kernel APIs consume
     * runtime M and intentionally do not reject values above this default.
     */
    inline constexpr int kDefaultNativeVNNIVerifierRowCapacity =
        kDefaultNativeVNNISmallMRows.back();

    /**
     * @brief Maximum rows processed by one batch-invariant GPU GEMV tile.
     *
     * This is an economy/workspace bound, never a supported-M limit. Larger
     * calls execute several grouped row tiles while retaining one production
     * operation and one persistent workspace binding.
     */
    inline constexpr int kDefaultNativeVNNIBatchInvariantTileRows = 128;

    /**
     * @brief Scratch budget for one ordered K-partition row tile.
     *
     * KPAR stores one FP32 partial per `(K block, row, output column)` in the
     * conservative workspace plan. Bounding that arena prevents vocabulary-
     * width and MTP-head projections from consuming the complete device while
     * allowing ordinary projections to expose substantially more row-level
     * parallelism than the historical sixteen-row tile.
     */
    inline constexpr size_t kDefaultNativeVNNIBatchInvariantScratchBytes =
        size_t{512} * 1024 * 1024;

    /**
     * @brief Select a total, geometry-aware row tile for batch-invariant GEMV.
     *
     * @param m Runtime row count; every positive value is accepted.
     * @param n Output width.
     * @param k Reduction width, rounded conservatively to 32-value blocks.
     * @return A positive row count no greater than `m` or the configured tile
     *         ceiling. Calls with two or more rows remain genuinely grouped
     *         whenever the scratch budget can hold two rows.
     */
    inline constexpr int nativeVNNIBatchInvariantTileRows(int m, int n, int k)
    {
        if (m <= 0 || n <= 0 || k <= 0)
            return 0;
        const size_t k_blocks =
            (static_cast<size_t>(k) + size_t{31}) / size_t{32};
        const size_t bytes_per_row =
            k_blocks * static_cast<size_t>(n) * sizeof(float);
        const size_t budget_rows = std::max(
            size_t{1},
            kDefaultNativeVNNIBatchInvariantScratchBytes /
                std::max(size_t{1}, bytes_per_row));
        const size_t grouped_floor = m > 1 ? size_t{2} : size_t{1};
        const size_t bounded_rows = std::max(
            grouped_floor,
            std::min(
                static_cast<size_t>(kDefaultNativeVNNIBatchInvariantTileRows),
                budget_rows));
        return std::min(m, static_cast<int>(bounded_rows));
    }

    /**
     * @brief Select the row tile for a grouped kernel with no K-partial arena.
     *
     * Direct and wide public-M1 policies accumulate the complete reduction in
     * registers and publish each verifier row directly.  Applying the KPAR
     * scratch budget to those kernels is both conceptually wrong and expensive:
     * a very wide LM head can be split into several launches even though it
     * allocates no per-row partials.  Keep the ordinary row ceiling so launch
     * geometry remains bounded, but otherwise let one launch cover the complete
     * small-M verifier batch.
     *
     * @param m Runtime row count; every positive value is accepted.
     * @return A positive row count no greater than @p m or the common tile
     *         ceiling, or zero for invalid input.
     */
    inline constexpr int nativeVNNIScratchlessBatchInvariantTileRows(int m)
    {
        return m > 0
                   ? std::min(m, kDefaultNativeVNNIBatchInvariantTileRows)
                   : 0;
    }

    /**
     * @brief Plan persistent verifier scratch independently of prompt length.
     *
     * A prepared GEMM object is shared by prefill, serial decode, and grouped
     * verification. Its large-M sizing request therefore describes the prompt
     * bucket, not the number of verifier rows that must coexist in one K-part
     * scratch tile. Reserving scratch for every prompt row can consume hundreds
     * of MiB even though the prefill kernel uses a different workspace.
     *
     * The persistent arena covers the complete default verifier capacity, or
     * the caller's smaller positive row count. Runtime-M implementations tile
     * larger verifier calls through this same arena, so this planning bound is
     * an economy decision rather than a semantic M limit.
     *
     * @param requested_m Workspace sizing request from the graph or test.
     * @param n Output width.
     * @param k Reduction width.
     * @return Geometry-bounded persistent verifier rows, or zero for invalid
     *         dimensions.
     */
    inline constexpr int nativeVNNIPersistentVerifierWorkspaceRows(
        int requested_m,
        int n,
        int k)
    {
        if (requested_m <= 0 || n <= 0 || k <= 0)
            return 0;
        const int planned_verifier_rows = std::min(
            requested_m,
            kDefaultNativeVNNIVerifierRowCapacity);
        return nativeVNNIBatchInvariantTileRows(
            planned_verifier_rows,
            n,
            k);
    }

    inline std::vector<int> defaultPrefillGraphBucketSizes()
    {
        return {kDefaultPrefillGraphBucketSizes.begin(),
                kDefaultPrefillGraphBucketSizes.end()};
    }

    inline std::vector<int> defaultNativeVNNIDispatchTrainingRows()
    {
        std::vector<int> rows;
        rows.reserve(kDefaultNativeVNNISmallMRows.size() +
                     kDefaultPrefillGraphBucketSizes.size());
        rows.insert(rows.end(),
                    kDefaultNativeVNNISmallMRows.begin(),
                    kDefaultNativeVNNISmallMRows.end());
        rows.insert(rows.end(),
                    kDefaultPrefillGraphBucketSizes.begin(),
                    kDefaultPrefillGraphBucketSizes.end());
        return rows;
    }

} // namespace llaminar2
