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
#include "../../utils/NativeVNNIEquivalenceInventory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
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

        /**
         * @test Prove row planning remains total across M, N, and K boundaries.
         *
         * This is deliberately a Cartesian host-only check. It includes
         * one-less/exact/one-more alignment points, verifier and tile limits,
         * graph-prefill sizes, and the largest positive API value. Every
         * positive geometry must return a usable row tile without overflow,
         * rejecting the historical assumption that only swept model shapes
         * need a valid generic decision.
         */
        TEST(Test__PrefillGraphBucketDefaults,
             IsTotalAcrossPositiveMAndGeometryBoundaries)
        {
            constexpr int maximum = std::numeric_limits<int>::max();
            constexpr std::array<int, 12> m_values = {
                1, 2, 15, 16, 17, 31,
                32, 127, 128, 129, 4096, maximum};
            constexpr std::array<int, 10> geometry_values = {
                1, 31, 32, 33, 127,
                128, 129, 4095, 4096, maximum};

            for (int m : m_values)
            {
                for (int n : geometry_values)
                {
                    for (int k : geometry_values)
                    {
                        const int rows =
                            nativeVNNIBatchInvariantTileRows(m, n, k);
                        EXPECT_GE(rows, 1)
                            << "M=" << m << " N=" << n << " K=" << k;
                        EXPECT_LE(rows, m)
                            << "M=" << m << " N=" << n << " K=" << k;
                        EXPECT_LE(
                            rows,
                            kDefaultNativeVNNIBatchInvariantTileRows)
                            << "M=" << m << " N=" << n << " K=" << k;
                        if (m > 1)
                        {
                            EXPECT_GE(rows, 2)
                                << "Positive grouped geometry lost grouped "
                                   "execution at M="
                                << m << " N=" << n << " K=" << k;
                        }
                    }
                }
            }
        }

        /**
         * @test Validate the finite GPU byte-equivalence witness construction.
         */
        TEST(Test__PrefillGraphBucketDefaults,
             EquivalenceInventoryExhaustsTileRowsAndBucketBoundaries)
        {
            const auto cases =
                test::nativeVNNIPrefillBucketEquivalenceCases();
            ASSERT_FALSE(cases.empty());

            int previous_active_rows = 0;
            for (const auto &test_case : cases)
            {
                EXPECT_GT(test_case.active_rows, previous_active_rows);
                EXPECT_GT(test_case.bucket_rows, test_case.active_rows);
                EXPECT_TRUE(std::binary_search(
                    kDefaultPrefillGraphBucketSizes.begin(),
                    kDefaultPrefillGraphBucketSizes.end(),
                    test_case.bucket_rows));
                previous_active_rows = test_case.active_rows;
            }

            for (int m =
                     kDefaultNativeVNNIVerifierRowCapacity + 1;
                 m <= 256;
                 ++m)
            {
                const bool is_exact_bucket = std::binary_search(
                    kDefaultPrefillGraphBucketSizes.begin(),
                    kDefaultPrefillGraphBucketSizes.end(),
                    m);
                const bool has_witness = std::any_of(
                    cases.begin(),
                    cases.end(),
                    [m](const auto &test_case)
                    {
                        return test_case.active_rows == m;
                    });
                EXPECT_EQ(has_witness, !is_exact_bucket)
                    << "M=" << m;
            }

            for (size_t index = 1;
                 index < kDefaultPrefillGraphBucketSizes.size();
                 ++index)
            {
                const int lower =
                    kDefaultPrefillGraphBucketSizes[index - 1];
                const int upper =
                    kDefaultPrefillGraphBucketSizes[index];
                if (upper <= 256)
                    continue;
                for (int active_rows : {lower + 1, upper - 1})
                {
                    if (active_rows == upper)
                        continue;
                    EXPECT_TRUE(std::any_of(
                        cases.begin(),
                        cases.end(),
                        [active_rows, upper](const auto &test_case)
                        {
                            return test_case.active_rows == active_rows &&
                                   test_case.bucket_rows == upper;
                        }))
                        << "Missing graph-bucket edge witness active M="
                        << active_rows << " bucket M=" << upper;
                }
            }
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
