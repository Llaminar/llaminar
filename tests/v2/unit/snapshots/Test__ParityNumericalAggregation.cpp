/**
 * @file Test__ParityNumericalAggregation.cpp
 * @brief Unit regressions for branch-aware MoE parity aggregation.
 */

#include <gtest/gtest.h>

#include "../../utils/ParityNumericalAggregation.h"

namespace llaminar2::test::parity
{
    TEST(Test__ParityNumericalAggregation, RoutingUsesDedicatedMetrics)
    {
        EXPECT_FALSE(parityStageContributesToLayerCosine(
            "MOE_ROUTING_INDICES", true));
        EXPECT_FALSE(parityStageContributesToLayerCosine(
            "MOE_ROUTING_WEIGHTS", true));
    }

    TEST(Test__ParityNumericalAggregation, RawExpertSumRequiresSameDiscreteBranch)
    {
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "MOE_EXPERT_OUTPUT", true));
        EXPECT_FALSE(parityStageContributesToLayerCosine(
            "MOE_EXPERT_OUTPUT", false));
    }

    TEST(Test__ParityNumericalAggregation, SemanticDownstreamStagesRemainAuthoritative)
    {
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "MOE_COMBINED_OUTPUT", false));
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "FFN_RESIDUAL", false));
        EXPECT_TRUE(parityStageContributesToLayerCosine(
            "LM_HEAD", false));
    }
} // namespace llaminar2::test::parity
