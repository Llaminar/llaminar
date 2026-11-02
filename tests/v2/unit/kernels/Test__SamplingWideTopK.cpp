/**
 * @file Test__SamplingWideTopK.cpp
 * @brief Device-free Top-K launch extents and read-only history arithmetic.
 *
 * The existing stochastic arena admits 8192 partial entries per row. Sweep all
 * supported wide widths and MTP row counts, including unseen vocabulary sizes,
 * to prove the selector remains within that budget and the 32-KiB shared limit.
 */
#include "kernels/common/SamplingWideTopKDevice.inl"
#include <gtest/gtest.h>
#include <limits>
#include <array>
#include <bit>
#include "backends/GenerationPenaltyHistory.h"

using namespace llaminar2;

/** @brief Every supported width and row count fits the existing admitted arena. */
TEST(SamplingWideTopK, AllWidthsAndRowsRespectAdmittedCapacity)
{
    for (int rows = 1; rows <= 16; ++rows)
    for (int k = 65; k <= sampling_math::kMaxTopK; ++k)
    for (int vocabulary : {65, 129, 248320, 300001, std::numeric_limits<int>::max()})
    {
        const auto plan = sampling_wide::geometry(vocabulary, rows, k, rows * 8192);
        if (vocabulary < k)
        {
            EXPECT_FALSE(plan);
            continue;
        }
        ASSERT_TRUE(plan);
        EXPECT_GT(plan->partial_blocks, 0);
        EXPECT_LE(plan->partial_blocks, sampling_wide::kPartialLists);
        EXPECT_LE(static_cast<size_t>(rows) * plan->partial_blocks * k,
                  static_cast<size_t>(rows) * 8192);
        EXPECT_LE(plan->shared_bytes, 32u * 1024u);
    }
}

/** @brief Missing/minimal storage and invalid public geometry fail before launch. */
TEST(SamplingWideTopK, RejectsInvalidOrInsufficientGeometry)
{
    EXPECT_FALSE(sampling_wide::geometry(248320, 0, 256, 8192));
    EXPECT_FALSE(sampling_wide::geometry(0, 16, 256, 131072));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 64, 131072));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 257, 131072));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 256, 4095));
    EXPECT_FALSE(sampling_wide::geometry(248320, 16, 256, 0));
    const auto minimum = sampling_wide::geometry(248320, 16, 256, 4096);
    ASSERT_TRUE(minimum);
    EXPECT_EQ(minimum->partial_blocks, 1);
}

/** @brief Real read-only candidate arithmetic matches the original serial law. */
TEST(SamplingWideTopK, HistoryCandidateReadPreservesRawRowsAndSerialPenaltyOrder)
{
    const std::array<float, 4> logits{-3.25f, 4.5f, -0.125f, 0.0f};
    const auto original = logits;
    const std::array<int, 4> counts{
        sampling_math::TokenPenaltyHistory::withPrompt(2), 0, 3,
        sampling_math::TokenPenaltyHistory::withPrompt(0)};
    const std::array<int, 4> tokens{1, 1, 2, 0};
    for (int enabled : {0, 1})
    for (int first_in_history : {0, 1})
    for (float repetition : {1.0f, 1.5f})
    for (bool speculative : {false, true})
    {
        MTPGreedyPenaltyPolicy policy{0.31f, 0.17f, repetition, first_in_history, enabled};
        const sampling_math::PenaltyLogitRead history{counts.data(), &policy, speculative ? tokens.data() : nullptr};
        for (int repeat = 0; repeat < 3; ++repeat)
        for (int row = 0; row < 4; ++row)
        for (int token = 0; token < 4; ++token)
        {
            float expected = logits[token];
            int count = sampling_math::TokenPenaltyHistory::generatedCount(counts[token]);
            if (speculative)
                for (int visible = first_in_history; visible <= row; ++visible)
                    count += tokens[visible] == token ? 1 : 0;
            if (enabled && (count > 0 || (repetition != 1.0f &&
                sampling_math::TokenPenaltyHistory::inPrompt(counts[token]))))
            {
                float additive = 0.0f;
                if (count > 0)
                    additive += policy.presence_penalty;
                additive += policy.frequency_penalty * static_cast<float>(count);
                if (repetition != 1.0f)
                    expected = expected < 0.0f ? expected * repetition : expected / repetition;
                expected -= additive;
            }
            const float observed = sampling_math::historyLogit<true>(logits[token], token, row, history);
            EXPECT_EQ(std::bit_cast<std::uint32_t>(observed), std::bit_cast<std::uint32_t>(expected));
            EXPECT_EQ(std::bit_cast<std::uint32_t>(sampling_math::historyLogit<false>(
                          logits[token], token, row, {})), std::bit_cast<std::uint32_t>(logits[token]));
        }
    }
    EXPECT_EQ(logits, original);
}

/** @brief Fixed and resident histories cannot fabricate or change their row provenance. */
TEST(SamplingWideTopK, HistoryFixedRowsRetainExactCapturedGeometry)
{
    int counts[4]{};
    int tokens[4]{};
    int active = 3;
    MTPGreedyPenaltyPolicy policy{};
    const auto fixed = GenerationPenaltyHistory::speculativeFixedRows(counts, &policy, tokens, 4);
    EXPECT_TRUE(fixed.admitsRows(4));
    EXPECT_FALSE(fixed.admitsRows(3));
    EXPECT_EQ(fixed.activeRows(), nullptr);
    const auto resident = GenerationPenaltyHistory::speculative(counts, &policy, tokens, &active);
    EXPECT_TRUE(resident.admitsRows(4));
    EXPECT_EQ(resident.activeRows(), &active);
    EXPECT_THROW(GenerationPenaltyHistory::speculative(counts, &policy, tokens, nullptr), std::invalid_argument);
    EXPECT_THROW(GenerationPenaltyHistory::speculativeFixedRows(counts, &policy, nullptr, 4), std::invalid_argument);
    EXPECT_THROW(GenerationPenaltyHistory::speculativeFixedRows(counts, &policy, tokens, 0), std::invalid_argument);
}
