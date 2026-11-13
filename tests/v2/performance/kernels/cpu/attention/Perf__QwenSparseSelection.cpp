/**
 * @file Perf__QwenSparseSelection.cpp
 * @brief Release CPU paired selector timing with immutable stored FP32 scores.
 *
 * Physical-core/NUMA/OpenMP budget belongs to canonical runner setup. This
 * harness never changes threads, ISA, affinity or initializes a GPU. A separate
 * observation call records compaction work after timed AB/BA with diagnostics
 * disabled. Exact output/source bytes are checked outside every timing window.
 */
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include "utils/CPUFeatures.h"
#include "../../../../integration/kernels/qwen4exp/QwenSparseSelectionCases.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
namespace llaminar2::qwen4exp
{
TEST(CPUQwenSparseSelectionPerf, PairedReleaseTiming)
{
    const char *rowsText = std::getenv("LLAMINAR_QSA_SELECT_M");
    const char *tokensText = std::getenv("LLAMINAR_QSA_SELECT_TOKENS");
    const char *repeatsText = std::getenv("LLAMINAR_QSA_SELECT_REPEATS");
    const char *distributionText = std::getenv("LLAMINAR_QSA_SELECT_DISTRIBUTION");
    if (!rowsText || !tokensText || !repeatsText || !distributionText)
        GTEST_SKIP() << "Explicit exact selector timing geometry/repeats/distribution required";
    const int rows = std::stoi(rowsText), tokens = std::stoi(tokensText), repeats = std::stoi(repeatsText);
    const int distributionId = std::stoi(distributionText);
    ASSERT_GT(rows, 0); ASSERT_LE(rows, 2048); ASSERT_GE(tokens, rows); ASSERT_LE(tokens, 262147);
    ASSERT_GT(repeats, 0); ASSERT_LE(repeats, 10000);
    ASSERT_GE(distributionId, 0); ASSERT_LT(distributionId, int(selection_test::distributions.size()));
    const int blocks = (tokens + 3) / 4;
    QwenSparseReplay replay{tokens - rows, rows, 0};
    auto scores = selection_test::scores(rows, blocks, selection_test::distributions[distributionId]);
    const auto original = scores;
    std::vector<std::int32_t> installed(std::size_t(rows) * QwenSparseSelected, -77), candidate(installed);
    std::vector<std::int32_t> installedCounts(rows), candidateCounts(rows);
    std::vector<QwenSparseSelectionObservation> observations(rows);
    QwenSparseIndexer a;
    a.replay = &replay; a.scores = scores.data(); a.selected = installed.data(); a.selected_count = installedCounts.data();
    a.row_capacity = rows; a.token_capacity = tokens; a.block_capacity = blocks;
    auto b = a; b.selected = candidate.data(); b.selected_count = candidateCounts.data();
    const auto verify = [&]()
    {
        ASSERT_EQ(installedCounts, candidateCounts); ASSERT_EQ(installed, candidate);
        ASSERT_EQ(std::memcmp(scores.data(), original.data(), scores.size() * sizeof(float)), 0);
        for (int row = 0; row < rows; ++row)
        {
            const auto expected = selection_test::oracle(scores.data() + std::size_t(row) * blocks,
                replay.prefix_tokens + row + 1);
            ASSERT_EQ(installedCounts[row], expected.size());
            ASSERT_TRUE(std::equal(expected.begin(), expected.end(), installed.begin() + std::size_t(row) * QwenSparseSelected));
        }
    };
    for (auto policy : selection_test::policies)
    {
        b.pool_selection = policy;
        for (int warm = 0; warm < 3; ++warm) { cpuQwenSparseSelectPools(a); cpuQwenSparseSelectPools(b); }
        verify(); ASSERT_FALSE(::testing::Test::HasFailure());
        const auto time = [&](const QwenSparseIndexer &p)
        {
            const auto begin = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < repeats; ++repeat) cpuQwenSparseSelectPools(p);
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() / repeats;
        };
        for (int pair = 0; pair < 4; ++pair)
        {
            const bool ab = pair % 2 == 0;
            const double first = ab ? time(a) : time(b), second = ab ? time(b) : time(a);
            std::cout << "QSA_SELECT_PAIR,backend=CPU,scope=stored_scores,order=" << (ab ? "AB" : "BA")
                      << ",M=" << rows << ",tokens=" << tokens << ",distribution=" << distributionId
                      << ",policy=" << int(policy) << ",workers=" << omp_get_max_threads()
                      << ",isa=" << int(activeISALevel()) << ",installed_ms=" << (ab ? first : second)
                      << ",candidate_ms=" << (ab ? second : first) << '\n';
        }
        verify(); ASSERT_FALSE(::testing::Test::HasFailure());
        b.selection_observations = observations.data(); cpuQwenSparseSelectPools(b); b.selection_observations = nullptr;
        QwenSparseSelectionObservation total{};
        for (const auto &row : observations)
        {
            total.score_elements += row.score_elements; total.compactions += row.compactions;
            total.compacted_keys += row.compacted_keys; total.radix_passes += row.radix_passes;
        }
        std::cout << "QSA_SELECT_WORK,policy=" << int(policy) << ",score_elements=" << total.score_elements
                  << ",compactions=" << total.compactions << ",compacted_keys=" << total.compacted_keys
                  << ",radix_passes=" << total.radix_passes << ",scope=diagnostic_not_timed_not_wire_bytes\n";
    }
}
} // namespace llaminar2::qwen4exp
