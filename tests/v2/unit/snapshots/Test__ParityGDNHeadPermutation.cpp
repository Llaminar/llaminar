/**
 * @file Test__ParityGDNHeadPermutation.cpp
 * @brief Unit regressions for unequal-head GDN parity checkpoint ordering.
 */

#include <gtest/gtest.h>
#include <array>
#include <numeric>

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

    /** @test Native-GGUF oracles compare every affected scalar/packed/value checkpoint without an HF reorder. */
    TEST(Test__ParityGDNHeadPermutation, NativeGGUFReferenceRetainsEveryStageAndRow)
    {
        const ParityGDNHeadConfig native{
            .n_k_heads = 16, .n_v_heads = 48, .d_state = 128, .is_moe = true,
            .reference_layout = ParityGDNReferenceLayout::NativeGGUF};
        auto grouped = native;
        grouped.reference_layout = ParityGDNReferenceLayout::HuggingFaceGrouped;
        EXPECT_FALSE(native.needsPermutation());
        EXPECT_TRUE(grouped.needsPermutation());
        const std::array<std::pair<const char *, size_t>, 7> checkpoints{{
            {"GDN_ALPHA", 48}, {"GDN_BETA", 48},
            {"GDN_Z_PROJECTION", 6144}, {"GDN_DELTA_RULE_OUTPUT", 6144},
            {"GDN_NORM_GATE_OUTPUT", 6144}, {"QKV_PROJECTION", 10240},
            {"GDN_CONV1D_OUTPUT", 10240}}};
        for (const size_t rows : {1u, 2u, 9u, 16u})
            for (const auto &[stage, width] : checkpoints)
            {
                std::vector<float> values(rows * width);
                std::iota(values.begin(), values.end(), 0.0f);
                const auto original = values;
                EXPECT_TRUE(applyParityGDNHeadPermutation(values.data(), values.size(), stage, native).empty())
                    << stage << " rows=" << rows;
                const auto hf = applyParityGDNHeadPermutation(values.data(), values.size(), stage, grouped);
                ASSERT_EQ(hf.size(), values.size());
                EXPECT_NE(hf, original) << stage << " must retain the genuine HF conversion";
                EXPECT_EQ(values, original) << "Diagnostic policy must never mutate production storage";
            }
    }

    /** @test Unknown reference order is fatal rather than silently treating arbitrary metadata as native. */
    TEST(Test__ParityGDNHeadPermutation, RejectsUnknownReferenceLayout)
    {
        auto invalid = kUnequalHeadMoE;
        invalid.reference_layout = static_cast<ParityGDNReferenceLayout>(255);
        const std::array<float, 4> values{};
        EXPECT_THROW(applyParityGDNHeadPermutation(values.data(), values.size(), "GDN_ALPHA", invalid),
                     std::invalid_argument);
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
