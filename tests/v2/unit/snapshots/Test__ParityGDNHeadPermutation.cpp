/**
 * @file Test__ParityGDNHeadPermutation.cpp
 * @brief Unit regressions for unequal-head GDN parity checkpoint ordering.
 */

#include <gtest/gtest.h>

#include "../../utils/ParityGDNHeadPermutation.h"

namespace llaminar2::test::parity
{
    namespace
    {
        const ParityGDNHeadConfig kUnequalHeadMoE{
            .n_k_heads = 2,
            .n_v_heads = 4,
            .d_state = 3,
            .is_moe = true,
        };
    }

    TEST(Test__ParityGDNHeadPermutation, MapsScalarAlphaAndBetaForEveryRow)
    {
        const std::vector<float> production{
            0.0f, 1.0f, 2.0f, 3.0f,
            10.0f, 11.0f, 12.0f, 13.0f,
        };
        const std::vector<float> expected{
            0.0f, 2.0f, 1.0f, 3.0f,
            10.0f, 12.0f, 11.0f, 13.0f,
        };

        EXPECT_EQ(
            applyParityGDNHeadPermutation(
                production.data(),
                production.size(),
                "GDN_ALPHA",
                kUnequalHeadMoE),
            expected);
        EXPECT_EQ(
            applyParityGDNHeadPermutation(
                production.data(),
                production.size(),
                "GDN_BETA",
                kUnequalHeadMoE),
            expected);
    }

    TEST(Test__ParityGDNHeadPermutation, MapsFullStateWidthWithoutMixingValues)
    {
        const std::vector<float> production{
            0.0f, 0.1f, 0.2f,
            1.0f, 1.1f, 1.2f,
            2.0f, 2.1f, 2.2f,
            3.0f, 3.1f, 3.2f,
        };
        const std::vector<float> expected{
            0.0f, 0.1f, 0.2f,
            2.0f, 2.1f, 2.2f,
            1.0f, 1.1f, 1.2f,
            3.0f, 3.1f, 3.2f,
        };

        EXPECT_EQ(
            applyParityGDNHeadPermutation(
                production.data(),
                production.size(),
                "GDN_Z_PROJECTION",
                kUnequalHeadMoE),
            expected);
    }

    TEST(Test__ParityGDNHeadPermutation, RejectsScalarTensorWithPartialRow)
    {
        const std::vector<float> production{0.0f, 1.0f, 2.0f};
        EXPECT_TRUE(applyParityGDNHeadPermutation(
                        production.data(),
                        production.size(),
                        "GDN_ALPHA",
                        kUnequalHeadMoE)
                        .empty());
    }
} // namespace llaminar2::test::parity
