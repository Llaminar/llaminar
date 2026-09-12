/**
 * @file Test__MoERouteConditionedProof.cpp
 * @brief Adversarial device-free certification of conditional HF expert equations.
 *
 * The fixture intentionally swaps a near-cutoff expert whose output is very
 * different. Legitimate route discontinuity must be distinguishable from bad
 * weights, bad input, corrupt arithmetic, stale banks, and missing evidence.
 */
#include <gtest/gtest.h>
#include "../../utils/MoERouteConditionedProof.h"
#include <sstream>

namespace llaminar2::test::parity
{
    /** @brief Small independent bank with a validated two-slot cutoff exchange. */
    class Test__MoERouteConditionedProof : public ::testing::Test
    {
    protected:
        std::vector<float> hf_router{.5f, .251f, .249f};
        std::vector<float> router{.5f, .249f, .251f};
        std::vector<float> hf_ids{0, 1}, ids{0, 2};
        std::vector<float> weights{.5f / .751f, .251f / .751f};
        std::vector<int> bank_ids{0, 1, 2};
        std::vector<float> bank{1, 0, 0, 1, 0, -1};
        std::vector<float> canonical, conditioned;
        MoERouteConditionedPrerequisites prerequisites;

        /** @brief Bind actual route and distribution proofs, not hand-set booleans. */
        void SetUp() override
        {
            prerequisites = {
                .boundary = compareMoERoutingBoundarySelections(
                    hf_ids.data(), ids.data(), ids.size(), hf_router.data(),
                    hf_router.size(), router.data(), router.size(), 2),
                .production_weights = proveMoENormalizedWeights(router, ids, weights, 2),
                .reference_weights = proveMoENormalizedWeights(hf_router, hf_ids, weights, 2),
                .router_kl = symmetricProbabilityKLDivergence(hf_router.data(), router.data(), 3, 3),
                .maximum_router_kl = .001,
                .hf_input_equivalent = true,
                .selected_set_changed = true,
            };
            canonical = sumMoEReferenceBank(bank, bank_ids, hf_ids, weights, 2, 2);
            conditioned = sumMoEReferenceBank(bank, bank_ids, ids, weights, 2, 2);
            ASSERT_TRUE(prerequisites.valid());
        }

        /** @brief Exercise the same typed authority used by every production backend. */
        MoERouteConditionedProof prove(const std::vector<float> &actual)
        {
            return proveMoERouteConditionedEquation(prerequisites, false,
                actual, conditioned, canonical, canonical, .98);
        }
    };

    TEST_F(Test__MoERouteConditionedProof, ValidCutoffPreservesOriginalHFFailure)
    {
        EXPECT_LT(measureMoEEquation(conditioned, canonical).cosine, .98);
        const auto proof = prove(conditioned);
        EXPECT_TRUE(proof.evaluated);
        EXPECT_TRUE(proof.equivalent);
        EXPECT_FALSE(proof.canonical_passed);
        EXPECT_DOUBLE_EQ(proof.conditioned.cosine, 1);
        EXPECT_DOUBLE_EQ(proof.canonical_reconstruction.relative_l2, 0);
    }

    TEST_F(Test__MoERouteConditionedProof, MissingPrerequisiteNeverCertifies)
    {
        const auto valid = prerequisites;
        for (int defect = 0; defect < 8; ++defect)
        {
            prerequisites = valid;
            switch (defect)
            {
                case 0: prerequisites.boundary.evaluated = false; break;
                case 1: prerequisites.boundary.equivalent = false; break;
                case 2: prerequisites.production_weights.valid = false; break;
                case 3: prerequisites.reference_weights.valid = false; break;
                case 4: prerequisites.hf_input_equivalent = false; break;
                case 5: prerequisites.selected_set_changed = false; break;
                case 6: prerequisites.router_kl = .1; break;
                case 7: prerequisites.router_kl = std::numeric_limits<double>::quiet_NaN(); break;
            }
            EXPECT_FALSE(prove(conditioned).equivalent) << defect;
        }
    }

    TEST_F(Test__MoERouteConditionedProof, CorruptExpertArithmeticAndAmplitudeFail)
    {
        auto doubled = conditioned;
        for (float &value : doubled) value *= 2;
        EXPECT_NEAR(measureMoEEquation(doubled, conditioned).cosine, 1, 1e-12);
        EXPECT_FALSE(prove(doubled).equivalent);
        EXPECT_FALSE(prove(canonical).equivalent);
        auto corrupt = conditioned;
        corrupt[0] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(prove(corrupt).evaluated);
        EXPECT_FALSE(prove({}).evaluated);
        auto wrong_bank = canonical;
        wrong_bank[0] += .01f;
        EXPECT_FALSE(proveMoERouteConditionedEquation(prerequisites, true,
            conditioned, conditioned, canonical, wrong_bank, .98).equivalent);
    }

    TEST_F(Test__MoERouteConditionedProof, WeightNormalizationRejectsCorruption)
    {
        EXPECT_FALSE(proveMoENormalizedWeights(router, ids, std::vector<float>{.5f, .5f}, 2).valid);
        EXPECT_FALSE(proveMoENormalizedWeights(router, std::vector<float>{0, 0}, weights, 2).valid);
        EXPECT_FALSE(proveMoENormalizedWeights(router, std::vector<float>{0, 3}, weights, 2).valid);
        EXPECT_FALSE(proveMoENormalizedWeights(router, std::vector<float>{0, .5f}, weights, 2).valid);
        EXPECT_FALSE(proveMoENormalizedWeights(router, ids, {}, 2).valid);
        EXPECT_FALSE(proveMoENormalizedWeights(router, ids, weights, 0).valid);
        router[0] = std::numeric_limits<float>::infinity();
        EXPECT_FALSE(proveMoENormalizedWeights(router, ids, weights, 2).valid);
    }

    TEST_F(Test__MoERouteConditionedProof, BankGeometryIdentityAndZeroAreFailClosed)
    {
        EXPECT_TRUE(sumMoEReferenceBank({}, bank_ids, ids, weights, 2, 2).empty());
        EXPECT_TRUE(sumMoEReferenceBank(bank, std::vector<int>{0, 2, 1}, ids, weights, 2, 2).empty());
        EXPECT_TRUE(sumMoEReferenceBank(bank, std::vector<int>{0, 1, 1}, ids, weights, 2, 2).empty());
        EXPECT_TRUE(sumMoEReferenceBank(bank, bank_ids, std::vector<float>{0, 4}, weights, 2, 2).empty());
        EXPECT_TRUE(sumMoEReferenceBank(bank, bank_ids, ids, weights, 2, 3).empty());
        const std::vector<float> zero(2, 0);
        EXPECT_DOUBLE_EQ(measureMoEEquation(zero, zero).cosine, 1);
        EXPECT_DOUBLE_EQ(measureMoEEquation(zero, zero).relative_l2, 0);
        EXPECT_FALSE(prove(zero).equivalent);
    }

    TEST_F(Test__MoERouteConditionedProof, CSVRetainsBothVerdictsAndStableColumns)
    {
        std::ostringstream header, canonical_row, conditional_row;
        writeMoERouteProofCSVHeader(header);
        writeMoERouteProofCSV(canonical_row, std::nullopt, true);
        writeMoERouteProofCSV(conditional_row, prove(conditioned), true);
        const auto columns = [](const std::string &text)
            { return 1 + std::count(text.begin(), text.end(), ','); };
        EXPECT_EQ(columns(header.str()), 9);
        EXPECT_EQ(columns(canonical_row.str()), columns(header.str()));
        EXPECT_EQ(columns(conditional_row.str()), columns(header.str()));
        EXPECT_TRUE(conditional_row.str().starts_with("route_conditioned_hf,0,1,1,"));
        EXPECT_TRUE(canonical_row.str().starts_with("canonical_hf,1,"));
    }

    TEST_F(Test__MoERouteConditionedProof, BoundedSuffixAuthenticatesEveryStageAndDependency)
    {
        const std::vector<float> fc{.1f, .2f}, attention{.2f, -.1f}, shared{.05f, .1f};
        const MoEHFSuffixOperands operands{fc, attention, shared};
        const MoEReferenceNormalization normalization{{1.f, .75f}, 1e-6f};
        const auto original = reconstructMoEHFSuffix(canonical, operands, normalization);
        const auto selected = reconstructMoEHFSuffix(conditioned, operands, normalization);
        ASSERT_TRUE(original);
        ASSERT_TRUE(selected);
        const MoESuffixInputProof inputs{MoEHFInputVerdict::Equivalent,
            MoEHFInputVerdict::Equivalent, MoEHFInputVerdict::Equivalent};
        const auto certify = [&](const MoESuffixRows &actual, const MoESuffixRows &hf,
                                 const MoESuffixInputProof &input_proof, const MoERouteConditionedProof &route)
        {
            return proveMoERouteConditionedSuffix(prerequisites, route, input_proof,
                actual, hf, {false, false, false}, original, selected, .98);
        };
        const auto valid = certify(*selected, *original, inputs, prove(conditioned));
        for (size_t i = 0; i < valid.size(); ++i)
        {
            EXPECT_TRUE(valid[i].equivalent);
            EXPECT_FALSE(valid[i].canonical_passed);
            EXPECT_EQ(valid[i].suffix_stage, static_cast<MoESuffixStage>(i));
            std::ostringstream csv;
            writeMoERouteProofCSV(csv, valid[i], true);
            EXPECT_TRUE(csv.str().starts_with("route_conditioned_hf_suffix,0,1,1,"));
        }
        // Corrupt each stage independently. Later arithmetic may look exactly
        // right, but it cannot erase an earlier failed dependency or HF bank.
        for (size_t bad = 0; bad < 3; ++bad)
        {
            auto actual = *selected;
            for (auto &value : actual[bad]) value *= 2;
            const auto corrupt = certify(actual, *original, inputs, prove(conditioned));
            auto stale_hf = *original;
            stale_hf[bad][0] += .01f;
            const auto stale = certify(*selected, stale_hf, inputs, prove(conditioned));
            for (size_t i = bad; i < 3; ++i)
            {
                EXPECT_FALSE(corrupt[i].equivalent) << bad << ':' << i;
                EXPECT_FALSE(stale[i].equivalent) << bad << ':' << i;
            }
            auto missing = inputs;
            if (bad == 0) missing.fc = MoEHFInputVerdict::Failed;
            if (bad == 1) missing.attention = MoEHFInputVerdict::Failed;
            if (bad == 2) missing.gated_shared = MoEHFInputVerdict::Failed;
            for (const auto &proof : certify(*selected, *original, missing, prove(conditioned)))
                EXPECT_FALSE(proof.equivalent);
        }
        for (const auto &proof : certify(*selected, *original, inputs, prove(canonical)))
            EXPECT_FALSE(proof.equivalent);
    }

    TEST_F(Test__MoERouteConditionedProof, SuffixRejectsMalformedAndNonfiniteEquations)
    {
        const std::vector<float> fc{.1f, .2f}, attention{.2f, -.1f}, shared{.05f, .1f};
        const MoEHFSuffixOperands operands{fc, attention, shared};
        const MoEReferenceNormalization normalization{{1.f, .75f}, 1e-6f};
        for (int bad = 0; bad < 7; ++bad)
        {
            auto norm = normalization;
            auto routed = conditioned;
            auto input = operands;
            switch (bad)
            {
                case 0: norm.scale.clear(); break;
                case 1: norm.epsilon = 0; break;
                case 2: norm.epsilon = std::numeric_limits<float>::quiet_NaN(); break;
                case 3: norm.scale[0] = std::numeric_limits<float>::infinity(); break;
                case 4: routed[0] = std::numeric_limits<float>::quiet_NaN(); break;
                case 5: input.fc = {}; break;
                case 6: routed.push_back(1); break;
            }
            EXPECT_FALSE(reconstructMoEHFSuffix(routed, input, norm)) << bad;
        }
        // Row-local RMS normalization must not reduce across grouped rows.
        const std::vector<float> rows{1, 2, 10, 20}, zero(4, 0);
        const auto grouped = reconstructMoEHFSuffix(rows, {zero, zero, zero}, normalization);
        ASSERT_TRUE(grouped);
        for (size_t row = 0; row < 2; ++row)
        {
            const auto serial = reconstructMoEHFSuffix(std::span(rows).subspan(row * 2, 2),
                {std::span(zero).first(2), std::span(zero).first(2), std::span(zero).first(2)}, normalization);
            ASSERT_TRUE(serial);
            for (size_t stage = 0; stage < 3; ++stage)
                for (size_t column = 0; column < 2; ++column)
                    EXPECT_EQ((*grouped)[stage][row * 2 + column], (*serial)[stage][column]);
        }
    }
}
