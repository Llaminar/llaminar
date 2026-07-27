/**
 * @file Test__PrefillGraphBucketDefaults.cpp
 * @brief Host-only regressions for NativeVNNI row-regime and tile planning.
 *
 * These tests intentionally perform no GPU work. They pin the shared planning
 * contract used before CUDA or ROCm workspace allocation so every positive M
 * remains dispatchable, speculative verifier calls remain genuinely grouped,
 * and large ordinary projections expose more row parallelism than the former
 * fixed sixteen-row implementation.
 */

#include "utils/PrefillGraphBucketDefaults.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>

namespace llaminar2
{
    namespace
    {
        TEST(Test__PrefillGraphBucketDefaults, RejectsInvalidGeometry)
        {
            EXPECT_EQ(nativeVNNIBatchInvariantTileRows(0, 4096, 4096), 0);
            EXPECT_EQ(nativeVNNIBatchInvariantTileRows(1, 0, 4096), 0);
            EXPECT_EQ(nativeVNNIBatchInvariantTileRows(1, 4096, -1), 0);
        }

        TEST(Test__PrefillGraphBucketDefaults, CoversDecodeAndEveryGroupedM)
        {
            constexpr int n = 4096;
            constexpr int k = 4096;

            EXPECT_EQ(nativeVNNIBatchInvariantTileRows(1, n, k), 1);
            for (int m = 2; m <= 16384; ++m)
            {
                const int rows = nativeVNNIBatchInvariantTileRows(m, n, k);
                EXPECT_GE(rows, 2) << "M=" << m;
                EXPECT_LE(rows, m) << "M=" << m;
                EXPECT_LE(rows, kDefaultNativeVNNIBatchInvariantTileRows)
                    << "M=" << m;
            }
        }

        TEST(Test__PrefillGraphBucketDefaults, ExpandsPastHistoricalMtpCapacity)
        {
            constexpr int m = 64;
            constexpr int n = 4096;
            constexpr int k = 4096;
            const int rows = nativeVNNIBatchInvariantTileRows(m, n, k);

            EXPECT_GT(rows, kDefaultNativeVNNIVerifierRowCapacity);
            EXPECT_EQ(rows, m);
        }

        TEST(Test__PrefillGraphBucketDefaults,
             PersistentVerifierWorkspaceDoesNotScaleWithPromptM)
        {
            constexpr int n = 17408;
            constexpr int k = 5120;

            const int verifier_rows =
                nativeVNNIPersistentVerifierWorkspaceRows(595, n, k);
            const int direct_prompt_rows =
                nativeVNNIBatchInvariantTileRows(595, n, k);

            EXPECT_EQ(verifier_rows, kDefaultNativeVNNIVerifierRowCapacity);
            EXPECT_GT(direct_prompt_rows, verifier_rows);
        }

        TEST(Test__PrefillGraphBucketDefaults,
             PersistentVerifierWorkspaceRemainsGroupedForWideGeometry)
        {
            constexpr int n = 248320;
            constexpr int k = 5120;

            const int rows =
                nativeVNNIPersistentVerifierWorkspaceRows(4096, n, k);

            EXPECT_GE(rows, 2);
            EXPECT_LE(rows, kDefaultNativeVNNIVerifierRowCapacity);
        }

        TEST(Test__PrefillGraphBucketDefaults, BoundsWideProjectionScratch)
        {
            constexpr int m = 4096;
            constexpr int n = 151936;
            constexpr int k = 8192;
            const int rows = nativeVNNIBatchInvariantTileRows(m, n, k);

            // The wide vocabulary projection cannot consume the ordinary
            // 128-row tile, but M>1 must still enter a grouped launch.
            EXPECT_EQ(rows, 3);
        }

        TEST(Test__PrefillGraphBucketDefaults, IsTotalAtPositiveIntExtremes)
        {
            constexpr int maximum = std::numeric_limits<int>::max();
            const int rows =
                nativeVNNIBatchInvariantTileRows(maximum, maximum, maximum);

            EXPECT_EQ(rows, 2);
        }

        TEST(Test__PrefillGraphBucketDefaults, TrainingRowsAreOrderedAndDisjoint)
        {
            const auto rows = defaultNativeVNNIDispatchTrainingRows();
            ASSERT_FALSE(rows.empty());
            EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end()));
            EXPECT_EQ(std::adjacent_find(rows.begin(), rows.end()), rows.end());
            EXPECT_EQ(rows.front(), 2);
            EXPECT_EQ(rows.back(), kDefaultPrefillGraphBucketSizes.back());
        }
    } // namespace
} // namespace llaminar2
