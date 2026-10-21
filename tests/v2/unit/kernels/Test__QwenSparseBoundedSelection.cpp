/**
 * @file Test__QwenSparseBoundedSelection.cpp
 * @brief Device-free exact candidate tests using actual CPU entrypoints.
 *
 * Independent numeric sort covers ties/radix boundaries, causal tails and dirty
 * replay. The scalar bounded helper is the actual CPU implementation, never a
 * test replacement for native selection. Worker changes are restored by RAII.
 */
#include "kernels/qwen4exp/QwenSparseBoundedSelection.h"
#include "../../integration/kernels/qwen4exp/QwenSparseSelectionCases.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <cstring>
namespace llaminar2::qwen4exp
{
namespace
{
struct Workers final
{
    int prior = omp_get_max_threads();
    ~Workers() { omp_set_num_threads(prior); }
};
struct Selection final
{
    int rows, capacity, blocks;
    QwenSparseReplay replay;
    std::vector<float> scores;
    std::vector<std::int32_t> selected, counts;
    std::vector<QwenSparseSelectionObservation> observations;
    QwenSparseIndexer p;
    Selection(int m, int tokens, int prefix, selection_test::Distribution distribution)
        : rows(m), capacity(tokens), blocks((tokens + 3) / 4), replay{prefix, m, 0},
          scores(selection_test::scores(m, blocks, distribution)),
          selected(std::size_t(m) * QwenSparseSelected, -77), counts(m, -77), observations(m)
    {
        p.replay = &replay; p.scores = scores.data(); p.selected = selected.data();
        p.selected_count = counts.data(); p.row_capacity = m; p.token_capacity = tokens; p.block_capacity = blocks;
    }
    void check()
    {
        for (int row = 0; row < rows; ++row)
        {
            const auto expected = row < replay.live_rows ? selection_test::oracle(
                scores.data() + std::size_t(row) * blocks, replay.prefix_tokens + row + 1) : std::vector<std::int32_t>{};
            ASSERT_EQ(counts[row], expected.size());
            for (std::size_t i = 0; i < expected.size(); ++i)
                ASSERT_EQ(selected[std::size_t(row) * QwenSparseSelected + i], expected[i]);
            for (std::size_t i = expected.size(); i < QwenSparseSelected; ++i)
                ASSERT_EQ(selected[std::size_t(row) * QwenSparseSelected + i], -77);
        }
    }
};
TEST(QwenSparseBoundedSelection, ExactTiesCommonPrefixesThresholdAndEveryTail)
{
    Workers workers; omp_set_num_threads(1);
    for (auto distribution : selection_test::distributions)
        for (int blocks : {511, 512, 513, 1025, 4097})
            for (int tail = 0; tail < 4; ++tail)
                for (auto policy : selection_test::policies)
                {
                    SCOPED_TRACE(int(distribution));
                    SCOPED_TRACE(blocks);
                    SCOPED_TRACE(tail);
                    SCOPED_TRACE(int(policy));
                    Selection s(1, blocks * 4 + tail, blocks * 4 + tail - 1, distribution);
                    const auto original = s.scores;
                    s.p.pool_selection = policy; s.p.selection_observations = s.observations.data();
                    cpuQwenSparseSelectPools(s.p); s.check();
                    EXPECT_EQ(std::memcmp(s.scores.data(), original.data(), original.size() * sizeof(float)), 0);
                    EXPECT_EQ(s.observations[0].score_elements, blocks > 512 ? std::uint64_t(blocks) : 0u);
                    if (blocks > 512) { EXPECT_GT(s.observations[0].compactions, 0u); }
                    else { EXPECT_EQ(s.observations[0].compactions, 0u); }
                }
}
TEST(QwenSparseBoundedSelection, GroupedSerialAndRetainedLargeSmallEmpty)
{
    Workers workers;
    for (int threads : {1, 3})
    {
        omp_set_num_threads(threads);
        for (auto policy : selection_test::policies)
        {
            Selection s(33, 8199, 8199 - 33, selection_test::Distribution::Hash);
            s.p.pool_selection = policy;
            for (int repeat = 0; repeat < 20; ++repeat)
            {
                s.replay.live_rows = repeat % 4 == 0 ? 0 : repeat % 4 == 1 ? 1 : repeat % 4 == 2 ? 9 : 33;
                s.replay.prefix_tokens = repeat % 3 == 0 ? 2048 : 8199 - 33;
                std::fill(s.selected.begin(), s.selected.end(), -77);
                cpuQwenSparseSelectPools(s.p); s.check();
                for (int row = 0; row < s.replay.live_rows; ++row)
                {
                    auto p = s.p; QwenSparseReplay r{s.replay.prefix_tokens + row, 1, 0};
                    std::array<std::int32_t, QwenSparseSelected> result; result.fill(-77);
                    int count = -1; p.replay = &r; p.row_capacity = 1;
                    p.scores += std::size_t(row) * s.blocks; p.selected = result.data(); p.selected_count = &count;
                    cpuQwenSparseSelectPools(p);
                    ASSERT_EQ(count, s.counts[row]);
                    EXPECT_EQ(std::memcmp(result.data(), s.selected.data() + std::size_t(row) * QwenSparseSelected,
                        sizeof(result)), 0);
                }
            }
        }
    }
}
TEST(QwenSparseBoundedSelection, StaticAndFiniteRejectionBeforeWritesUnusedScoresUntouched)
{
    Workers workers; omp_set_num_threads(1);
    Selection s(2, 4099, 4097, selection_test::Distribution::Equal);
    s.p.pool_selection = QwenSparsePoolSelection::Bounded256_512;
    auto p = s.p; p.pool_selection = static_cast<QwenSparsePoolSelection>(255);
    EXPECT_THROW(cpuQwenSparseSelectPools(p), std::invalid_argument);
    p = s.p; p.selected_count = p.selected; EXPECT_THROW(cpuQwenSparseSelectPools(p), std::invalid_argument);
    p = s.p; p.block_capacity = 1; EXPECT_THROW(cpuQwenSparseSelectPools(p), std::invalid_argument);
    for (float invalid : {-1.0f, INFINITY, NAN})
    {
        s.scores[0] = invalid;
        EXPECT_THROW(cpuQwenSparseSelectPools(s.p), std::invalid_argument);
        EXPECT_TRUE(std::all_of(s.selected.begin(), s.selected.end(), [](auto v) { return v == -77; }));
        EXPECT_EQ(s.counts[0], -77); EXPECT_EQ(s.counts[1], -77);
    }
    s.scores[0] = 0.0f;
    const int laterLivePools = (s.replay.prefix_tokens + 2) / QwenSparsePool;
    ASSERT_EQ(laterLivePools, 1024);
    const auto lastLive = std::size_t(s.blocks) + laterLivePools - 1;
    const auto firstUnused = std::size_t(s.blocks) + laterLivePools;
    s.scores[lastLive] = NAN; // The last consumed score of row1 must fail before row0 writes.
    EXPECT_THROW(cpuQwenSparseSelectPools(s.p), std::invalid_argument);
    EXPECT_TRUE(std::all_of(s.selected.begin(), s.selected.end(), [](auto v) { return v == -77; }));
    EXPECT_EQ(s.counts[0], -77); EXPECT_EQ(s.counts[1], -77);
    s.scores[lastLive] = 0.0f;
    ASSERT_LT(firstUnused, s.scores.size());
    s.scores[firstUnused] = NAN; // Allocated suffix is neither validated nor selected.
    EXPECT_NO_THROW(cpuQwenSparseSelectPools(s.p));
    s.check();
    EXPECT_TRUE(std::isnan(s.scores[firstUnused]));
    p = s.p; p.scores = reinterpret_cast<float *>(UINTPTR_MAX - 3u);
    EXPECT_THROW(validateQwenSparseSelection(p), std::invalid_argument);
    s.replay = {2048, 1, 0}; // 512 complete pools plus one tail: scores are NOT consumed.
    EXPECT_NO_THROW(cpuQwenSparseSelectPools(s.p));
    EXPECT_EQ(s.counts[0], 2049); EXPECT_EQ(s.counts[1], 0);
    s.replay.live_rows = 0; EXPECT_NO_THROW(cpuQwenSparseSelectPools(s.p));
}
TEST(QwenSparseBoundedSelection, GeometrySharedBytesAndUniqueTotalKey)
{
    using namespace bounded_selection;
    EXPECT_EQ(sizeof(Storage<256, 512>), 21744u);
    EXPECT_EQ(sizeof(Storage<512, 512>), 25840u);
    EXPECT_EQ(sizeof(Storage<1024, 1024>), 42224u);
    EXPECT_LT(sizeof(Storage<1024, 1024>), 48u * 1024u);
    EXPECT_LT(sizeof(Storage<256, 512>), sizeof(Storage<512, 512>));
    EXPECT_GT(key(0x3f800000u, 0), key(0x3f800000u, 1));
    EXPECT_EQ(key(0x80000000u, 0), key(0u, 0));
    EXPECT_FALSE(validScoreBits(0x7f800000u)); EXPECT_FALSE(validScoreBits(0xbf800000u));
    EXPECT_LT(firstShift(key(0x3f800000u, 0), key(0x3f800000u, 1024)), 32);
}
} // namespace
} // namespace llaminar2::qwen4exp
