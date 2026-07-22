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
    inline constexpr std::array<int, 21> kDefaultPrefillGraphBucketSizes = {
        64, 128, 256, 384, 512, 544, 576, 600, 608, 640, 672,
        704, 736, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096};

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
