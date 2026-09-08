/**
 * @file Test__ROCmMoEGroupedPrefillDispatch.cpp
 * @brief CPU-only structural tests for generated ROCm grouped-MoE prefill policy.
 *
 * These tests deliberately perform no GPU work. They validate the generated
 * decision surface consumed when a ROCm graph is captured: every advertised
 * execution codebook, projection role, aspect-ratio bucket, and M anchor must
 * resolve to a compiled tile, while unknown inputs must fail instead of falling
 * back to a legacy default.
 */

#include <gtest/gtest.h>

#include "kernels/rocm/gemm/ROCmMoEGroupedPrefillDispatchGenerated.inc"
#include "kernels/rocm/gemm/ROCmMoEGroupedPrefillRoutePolicy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace
{
    using llaminar2::rocm::generated::ROCmMoEGroupedPrefillConfig;
    using llaminar2::rocm::generated::kROCmMoEGroupedPrefillEntries;
    using llaminar2::rocm::generated::kROCmMoEGroupedPrefillMAnchors;
    using llaminar2::rocm::generated::rocmMoEGroupedPrefillMAnchor;
    using llaminar2::rocm::generated::rocmMoEGroupedPrefillRatioExponent;
    using llaminar2::rocm::generated::selectROCmMoEGroupedPrefillGenerated;

    constexpr std::array<uint8_t, 16> kExecutionCodebooks = {
        0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19};

    /** Representative N:K dimensions for ratio exponents -3 through +3. */
    constexpr std::array<std::pair<int, int>, 7> kRatioShapes = {{
        {256, 2048},
        {256, 1024},
        {256, 512},
        {256, 256},
        {512, 256},
        {1024, 256},
        {2048, 256},
    }};

    bool isCompiledTile(const ROCmMoEGroupedPrefillConfig &config)
    {
        const bool valid_m =
            config.tile_m == 4 || config.tile_m == 8 ||
            config.tile_m == 12 || config.tile_m == 16;
        const bool valid_n =
            config.tile_n == 64 || config.tile_n == 128 ||
            config.tile_n == 256;
        return valid_m && valid_n;
    }
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, CompletePolicyCartesianProductResolves)
{
    constexpr size_t expected_entries =
        kExecutionCodebooks.size() * 2u * kRatioShapes.size() *
        (sizeof(kROCmMoEGroupedPrefillMAnchors) /
         sizeof(kROCmMoEGroupedPrefillMAnchors[0]));
    static_assert(
        sizeof(kROCmMoEGroupedPrefillEntries) /
                sizeof(kROCmMoEGroupedPrefillEntries[0]) ==
            expected_entries);

    for (uint8_t codebook : kExecutionCodebooks)
    {
        for (uint8_t role : {uint8_t{0}, uint8_t{1}})
        {
            for (size_t ratio_index = 0; ratio_index < kRatioShapes.size(); ++ratio_index)
            {
                const auto [n, k] = kRatioShapes[ratio_index];
                ASSERT_EQ(
                    rocmMoEGroupedPrefillRatioExponent(n, k),
                    static_cast<int>(ratio_index) - 3);
                for (uint16_t m : kROCmMoEGroupedPrefillMAnchors)
                {
                    ROCmMoEGroupedPrefillConfig config{};
                    ASSERT_TRUE(selectROCmMoEGroupedPrefillGenerated(
                        codebook, role, m, n, k, config))
                        << "codebook=" << static_cast<int>(codebook)
                        << " role=" << static_cast<int>(role)
                        << " ratio=" << static_cast<int>(ratio_index) - 3
                        << " M=" << m;
                    EXPECT_TRUE(isCompiledTile(config));
                }
            }
        }
    }
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, RatioBucketsUseNearestLog2Boundaries)
{
    // sqrt(2), sqrt(8), and sqrt(32) are the exact boundaries between
    // neighboring powers of two. Integer dimensions are checked through their
    // squares, avoiding floating-point rounding in graph-capture policy code.
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(141, 100), 0);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(142, 100), 1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(282, 100), 1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(283, 100), 2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(565, 100), 2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(566, 100), 3);

    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 141), 0);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 142), -1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 282), -1);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 283), -2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 565), -2);
    EXPECT_EQ(rocmMoEGroupedPrefillRatioExponent(100, 566), -3);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, NearestMAnchorTiesPreferSmallerBucket)
{
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(1), 12);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(14), 12);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(20), 16);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(28), 24);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(192), 128);
    EXPECT_EQ(rocmMoEGroupedPrefillMAnchor(4096), 256);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch, UnsupportedPolicyHardFailsWithZeroConfig)
{
    ROCmMoEGroupedPrefillConfig unknown_codebook{};
    EXPECT_FALSE(selectROCmMoEGroupedPrefillGenerated(
        /*codebook=*/1,
        /*role=*/0,
        /*m=*/32,
        /*n=*/512,
        /*k=*/2048,
        unknown_codebook));
    EXPECT_EQ(unknown_codebook.tile_m, 0);
    EXPECT_EQ(unknown_codebook.tile_n, 0);

    ROCmMoEGroupedPrefillConfig invalid_role{};
    EXPECT_FALSE(selectROCmMoEGroupedPrefillGenerated(
        /*codebook=*/13,
        /*role=*/2,
        /*m=*/32,
        /*n=*/512,
        /*k=*/2048,
        invalid_role));
    EXPECT_EQ(invalid_role.tile_m, 0);
    EXPECT_EQ(invalid_role.tile_n, 0);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch,
     Qwen35_122BSparseEndpointUsesMeasuredRouteOwnedBuckets)
{
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteKey;
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteStrategy;
    using llaminar2::rocm::selectROCmMoEGroupedPrefillRouteStrategy;

    ROCmMoEGroupedPrefillRouteKey key{
        .gateup_codebook = 19, // NativeVNNI Q8_0 execution id.
        .down_codebook = 19,
        .hidden_size = 3072,
        .expert_width = 1024,
        .expert_count = 256,
        .top_k = 1,
        .rows = 16,
    };
    for (const int rows : {16, 32})
    {
        key.rows = rows;
        const auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
        EXPECT_EQ(
            decision.strategy,
            ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
        EXPECT_TRUE(decision.exact);
    }

    // Adjacent untrained identities must not inherit the measured exception.
    key.rows = 64;
    auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    EXPECT_FALSE(decision.exact);

    key.rows = 16;
    key.top_k = 2;
    decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    EXPECT_FALSE(decision.exact);
}

TEST(Test__ROCmMoEGroupedPrefillDispatch,
     GenericRoutePolicyIsTotalAndRejectsMalformedGeometry)
{
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteKey;
    using llaminar2::rocm::ROCmMoEGroupedPrefillRouteStrategy;
    using llaminar2::rocm::selectROCmMoEGroupedPrefillRouteStrategy;

    ROCmMoEGroupedPrefillRouteKey key{
        .gateup_codebook = 13,
        .down_codebook = 4,
        .hidden_size = 2048,
        .expert_width = 512,
        .expert_count = 256,
        .top_k = 8,
        .rows = 8,
    };
    auto decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::RouteOwned);
    EXPECT_FALSE(decision.exact);

    key.rows = 9;
    decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::ExpertTiled);
    EXPECT_FALSE(decision.exact);

    key.rows = 0;
    decision = selectROCmMoEGroupedPrefillRouteStrategy(key);
    EXPECT_EQ(
        decision.strategy,
        ROCmMoEGroupedPrefillRouteStrategy::Invalid);
    EXPECT_FALSE(decision.valid());
}
