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
     * @brief Small-M verifier/projection rows that share NativeVNNI dispatch
     * training with prefill buckets.
     */
    inline constexpr std::array<int, 3> kDefaultNativeVNNISmallMRows = {2, 3, 4};

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
