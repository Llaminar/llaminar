#pragma once

#include <array>
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
