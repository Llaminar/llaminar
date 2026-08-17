/**
 * @file Test__MoERoutingBoundary.cpp
 * @brief Device-free regressions for numerically unresolved MoE top-k cutoffs.
 */

#include <gtest/gtest.h>

#include "../../utils/MoERoutingBoundary.h"

#include <cmath>
#include <limits>

namespace llaminar2::test::parity
{
    TEST(Test__MoERoutingBoundary, ProbabilityKLConsumesPostSoftmaxRowsDirectly)
    {
        const float reference[] = {0.99f, 0.01f};
        const float production[] = {0.5f, 0.5f};

        const double observed = symmetricProbabilityKLDivergence(
            reference,
            production,
            2,
            2);
        const double expected = 0.5 * (
            0.99 * std::log(0.99 / 0.5) +
            0.01 * std::log(0.01 / 0.5) +
            0.5 * std::log(0.5 / 0.99) +
            0.5 * std::log(0.5 / 0.01));

        EXPECT_NEAR(observed, expected, 1.0e-7);
        EXPECT_GT(observed, 0.1)
            << "A second softmax would flatten these probability rows";
    }

    TEST(Test__MoERoutingBoundary, ProbabilityKLNormalizesAndReportsWorstRow)
    {
        const float reference[] = {
            0.2f,
            0.8f,
            2.0f,
            8.0f,
        };
        const float production[] = {
            0.2f,
            0.8f,
            5.0f,
            5.0f,
        };

        const double first_row = symmetricProbabilityKLDivergence(
            reference,
            production,
            2,
            2);
        const double second_row = symmetricProbabilityKLDivergence(
            reference + 2,
            production + 2,
            2,
            2);
        const double both_rows = symmetricProbabilityKLDivergence(
            reference,
            production,
            4,
            2);

        EXPECT_DOUBLE_EQ(first_row, 0.0);
        EXPECT_GT(second_row, 0.0);
        EXPECT_DOUBLE_EQ(both_rows, second_row);
    }

    TEST(Test__MoERoutingBoundary, ProbabilityKLRejectsMalformedDistributions)
    {
        const float valid[] = {0.25f, 0.75f};
        const float negative[] = {-0.25f, 1.25f};
        const float zero_sum[] = {0.0f, 0.0f};
        const float non_finite[] = {
            0.25f,
            std::numeric_limits<float>::quiet_NaN(),
        };

        EXPECT_TRUE(std::isinf(symmetricProbabilityKLDivergence(
            nullptr,
            valid,
            2,
            2)));
        EXPECT_TRUE(std::isinf(symmetricProbabilityKLDivergence(
            valid,
            valid,
            2,
            3)));
        EXPECT_TRUE(std::isinf(symmetricProbabilityKLDivergence(
            valid,
            negative,
            2,
            2)));
        EXPECT_TRUE(std::isinf(symmetricProbabilityKLDivergence(
            valid,
            zero_sum,
            2,
            2)));
        EXPECT_TRUE(std::isinf(symmetricProbabilityKLDivergence(
            valid,
            non_finite,
            2,
            2)));
    }

    TEST(Test__MoERoutingBoundary, ExactTopKSelectionPasses)
    {
        const float reference_indices[] = {0.0f, 1.0f};
        const float production_indices[] = {0.0f, 1.0f};
        const float reference_router[] = {0.9f, 0.8f, 0.7f, 0.1f};
        const float production_router[] = {0.9001f, 0.7999f, 0.7001f, 0.1f};

        const auto result = compareMoERoutingBoundarySelections(
            reference_indices,
            production_indices,
            2,
            reference_router,
            4,
            production_router,
            4,
            2);

        EXPECT_TRUE(result.evaluated);
        EXPECT_TRUE(result.equivalent);
        EXPECT_DOUBLE_EQ(result.maximum_boundary_gap, 0.0);
        EXPECT_GT(result.maximum_error_limit, 0.0);
    }

    TEST(Test__MoERoutingBoundary, CutoffSwapWithinMeasuredErrorPasses)
    {
        const float reference_indices[] = {0.0f, 1.0f};
        const float production_indices[] = {0.0f, 2.0f};
        const float reference_router[] = {
            0.9f,
            0.8000003f,
            0.8f,
            0.1f,
        };
        const float production_router[] = {0.90001f, 0.7998f, 0.8001f, 0.1f};

        const auto result = compareMoERoutingBoundarySelections(
            reference_indices,
            production_indices,
            2,
            reference_router,
            4,
            production_router,
            4,
            2);

        EXPECT_TRUE(result.evaluated);
        EXPECT_TRUE(result.equivalent);
        EXPECT_GT(result.maximum_boundary_gap, 0.0);
        EXPECT_LE(
            result.maximum_boundary_gap,
            result.maximum_error_limit);
    }

    TEST(Test__MoERoutingBoundary, DifferentTop1ExpertFailsClosed)
    {
        const float reference_indices[] = {0.0f, 1.0f};
        const float production_indices[] = {1.0f, 0.0f};
        const float reference_router[] = {0.9f, 0.8f, 0.7f, 0.1f};
        const float production_router[] = {0.9f, 0.8f, 0.7f, 0.1f};

        const auto result = compareMoERoutingBoundarySelections(
            reference_indices,
            production_indices,
            2,
            reference_router,
            4,
            production_router,
            4,
            2);

        EXPECT_TRUE(result.evaluated);
        EXPECT_FALSE(result.equivalent);
    }

    TEST(Test__MoERoutingBoundary, IndicesMustSelectTheirOwnLiveRouterTopK)
    {
        const float reference_indices[] = {0.0f, 1.0f};
        const float production_indices[] = {0.0f, 3.0f};
        const float reference_router[] = {0.9f, 0.8f, 0.7f, 0.1f};
        const float production_router[] = {0.9f, 0.8f, 0.7f, 0.1f};

        const auto result = compareMoERoutingBoundarySelections(
            reference_indices,
            production_indices,
            2,
            reference_router,
            4,
            production_router,
            4,
            2);

        EXPECT_TRUE(result.evaluated);
        EXPECT_FALSE(result.equivalent);
    }

    TEST(Test__MoERoutingBoundary, NonIntegralIndicesAndNonFiniteScoresFailClosed)
    {
        const float reference_indices[] = {0.0f, 1.0f};
        const float non_integral_indices[] = {0.0f, 1.5f};
        const float reference_router[] = {0.9f, 0.8f, 0.7f, 0.1f};
        const float non_finite_router[] = {
            0.9f,
            0.8f,
            std::numeric_limits<float>::quiet_NaN(),
            0.1f,
        };

        const auto invalid_indices = compareMoERoutingBoundarySelections(
            reference_indices,
            non_integral_indices,
            2,
            reference_router,
            4,
            reference_router,
            4,
            2);
        const auto invalid_scores = compareMoERoutingBoundarySelections(
            reference_indices,
            reference_indices,
            2,
            reference_router,
            4,
            non_finite_router,
            4,
            2);

        EXPECT_TRUE(invalid_indices.evaluated);
        EXPECT_FALSE(invalid_indices.equivalent);
        EXPECT_TRUE(invalid_scores.evaluated);
        EXPECT_FALSE(invalid_scores.equivalent);
    }
} // namespace llaminar2::test::parity
