/**
 * @file Test__DecodeExpertHistogram.cpp
 * @brief Unit tests for DecodeExpertHistogram
 */

#include <gtest/gtest.h>
#include "execution/moe/DecodeExpertHistogram.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

using namespace llaminar2;

// ── Helpers ───────────────────────────────────────────

static DecodeExpertHistogramConfig makeConfig(
    int num_layers, int num_experts, int top_k, int window_size,
    int num_sockets = 2)
{
    DecodeExpertHistogramConfig cfg;
    cfg.num_layers = num_layers;
    cfg.num_experts = num_experts;
    cfg.top_k = top_k;
    cfg.window_size = window_size;

    for (int s = 0; s < num_sockets; ++s)
        cfg.sockets.push_back(DeviceId(DeviceType::CPU, s));

    // Default: round-robin expert-to-socket
    cfg.expert_to_socket.resize(num_experts);
    for (int e = 0; e < num_experts; ++e)
        cfg.expert_to_socket[e] = e % num_sockets;

    return cfg;
}

// ── Tests ─────────────────────────────────────────────

TEST(Test__DecodeExpertHistogram, Construction)
{
    auto cfg = makeConfig(4, 64, 8, 256);
    DecodeExpertHistogram hist(cfg);

    EXPECT_EQ(hist.config().num_layers, 4);
    EXPECT_EQ(hist.config().num_experts, 64);
    EXPECT_EQ(hist.config().top_k, 8);
    EXPECT_EQ(hist.config().window_size, 256);

    // All counts start at zero
    for (int l = 0; l < 4; ++l)
    {
        for (int e = 0; e < 64; ++e)
        {
            EXPECT_EQ(hist.activationCount(l, e), 0u);
            EXPECT_FLOAT_EQ(hist.weightedActivation(l, e), 0.0f);
        }
    }
    EXPECT_EQ(hist.windowTokenCount(), 0u);
    EXPECT_EQ(hist.windowGeneration(), 0u);
    EXPECT_FALSE(hist.windowFull());
}

TEST(Test__DecodeExpertHistogram, SingleRecord)
{
    auto cfg = makeConfig(1, 8, 2, 256);
    DecodeExpertHistogram hist(cfg);

    int indices[] = {3, 7};
    float weights[] = {0.6f, 0.4f};
    hist.record(0, indices, weights, 2);

    EXPECT_EQ(hist.activationCount(0, 3), 1u);
    EXPECT_EQ(hist.activationCount(0, 7), 1u);
    EXPECT_EQ(hist.activationCount(0, 0), 0u);
    EXPECT_EQ(hist.activationCount(0, 1), 0u);

    EXPECT_FLOAT_EQ(hist.weightedActivation(0, 3), 0.6f);
    EXPECT_FLOAT_EQ(hist.weightedActivation(0, 7), 0.4f);
    EXPECT_EQ(hist.windowTokenCount(), 1u);
}

TEST(Test__DecodeExpertHistogram, MultipleRecords_SameLayer)
{
    auto cfg = makeConfig(1, 8, 2, 256);
    DecodeExpertHistogram hist(cfg);

    // Token 1: experts 0, 1
    int idx1[] = {0, 1};
    float w1[] = {0.7f, 0.3f};
    hist.record(0, idx1, w1, 2);

    // Token 2: experts 0, 2
    int idx2[] = {0, 2};
    float w2[] = {0.5f, 0.5f};
    hist.record(0, idx2, w2, 2);

    // Token 3: experts 0, 3
    int idx3[] = {0, 3};
    float w3[] = {0.9f, 0.1f};
    hist.record(0, idx3, w3, 2);

    EXPECT_EQ(hist.activationCount(0, 0), 3u); // Expert 0 selected in all 3 tokens
    EXPECT_EQ(hist.activationCount(0, 1), 1u);
    EXPECT_EQ(hist.activationCount(0, 2), 1u);
    EXPECT_EQ(hist.activationCount(0, 3), 1u);

    EXPECT_NEAR(hist.weightedActivation(0, 0), 0.7f + 0.5f + 0.9f, 1e-5f);
    EXPECT_EQ(hist.windowTokenCount(), 3u);
}

TEST(Test__DecodeExpertHistogram, MultipleRecords_DifferentLayers)
{
    auto cfg = makeConfig(3, 8, 2, 256);
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0, 1};
    float w[] = {0.5f, 0.5f};

    // Simulate 3 complete decode tokens (all layers per token)
    for (int token = 0; token < 3; ++token)
        for (int l = 0; l < 3; ++l)
            hist.record(l, idx, w, 2);

    // Each layer sees 3 tokens
    EXPECT_EQ(hist.activationCount(0, 0), 3u);
    EXPECT_EQ(hist.activationCount(1, 0), 3u);
    EXPECT_EQ(hist.activationCount(2, 0), 3u);

    // Window counts actual decode tokens (incremented once per last-layer record)
    EXPECT_EQ(hist.windowTokenCount(), 3u);
}

TEST(Test__DecodeExpertHistogram, SocketLoads_BalancedPlacement)
{
    // 8 experts, round-robin across 2 sockets:
    //   socket 0: experts 0,2,4,6
    //   socket 1: experts 1,3,5,7
    auto cfg = makeConfig(1, 8, 2, 256, 2);
    DecodeExpertHistogram hist(cfg);

    // Route equally to one expert per socket
    int idx1[] = {0, 1};
    float w1[] = {0.5f, 0.5f};
    for (int i = 0; i < 100; ++i)
        hist.record(0, idx1, w1, 2);

    auto loads = hist.socketLoads(0);
    ASSERT_EQ(loads.size(), 2u);
    EXPECT_EQ(loads[0], 100u); // Expert 0 → socket 0
    EXPECT_EQ(loads[1], 100u); // Expert 1 → socket 1
    EXPECT_FLOAT_EQ(hist.socketImbalanceRatio(0), 1.0f);
}

TEST(Test__DecodeExpertHistogram, SocketLoads_SkewedRouting)
{
    // All routing hits experts on socket 0
    auto cfg = makeConfig(1, 8, 2, 256, 2);
    DecodeExpertHistogram hist(cfg);

    // Experts 0 and 2 are both on socket 0 (round-robin: even → sock 0)
    int idx[] = {0, 2};
    float w[] = {0.5f, 0.5f};
    for (int i = 0; i < 50; ++i)
        hist.record(0, idx, w, 2);

    auto loads = hist.socketLoads(0);
    EXPECT_EQ(loads[0], 100u); // All on socket 0
    EXPECT_EQ(loads[1], 0u);   // Nothing on socket 1
    EXPECT_TRUE(std::isinf(hist.socketImbalanceRatio(0)));
}

TEST(Test__DecodeExpertHistogram, SocketImbalanceRatio_PerfectBalance)
{
    auto cfg = makeConfig(1, 4, 2, 256, 2);
    // socket 0: experts 0,2; socket 1: experts 1,3
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0, 1};
    float w[] = {0.5f, 0.5f};
    for (int i = 0; i < 100; ++i)
        hist.record(0, idx, w, 2);

    EXPECT_FLOAT_EQ(hist.socketImbalanceRatio(0), 1.0f);
}

TEST(Test__DecodeExpertHistogram, SocketImbalanceRatio_TotalSkew)
{
    auto cfg = makeConfig(1, 4, 1, 256, 2);
    DecodeExpertHistogram hist(cfg);

    // Only route to expert 0 (socket 0)
    int idx[] = {0};
    float w[] = {1.0f};
    for (int i = 0; i < 100; ++i)
        hist.record(0, idx, w, 1);

    // Socket 0 has all load, socket 1 has zero → infinity
    EXPECT_TRUE(std::isinf(hist.socketImbalanceRatio(0)));
}

TEST(Test__DecodeExpertHistogram, WindowFull_ExactSize)
{
    auto cfg = makeConfig(1, 4, 1, 10);
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0};
    float w[] = {1.0f};

    for (int i = 0; i < 9; ++i)
    {
        hist.record(0, idx, w, 1);
        EXPECT_FALSE(hist.windowFull());
    }
    hist.record(0, idx, w, 1);
    EXPECT_TRUE(hist.windowFull());
    EXPECT_EQ(hist.windowTokenCount(), 10u);
}

TEST(Test__DecodeExpertHistogram, WindowReset_ClearsCounters)
{
    auto cfg = makeConfig(2, 4, 2, 256, 2);
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0, 1};
    float w[] = {0.6f, 0.4f};
    // Simulate 50 complete decode tokens (both layers per token)
    for (int i = 0; i < 50; ++i)
    {
        hist.record(0, idx, w, 2);
        hist.record(1, idx, w, 2);
    }

    EXPECT_EQ(hist.windowGeneration(), 0u);
    EXPECT_GT(hist.activationCount(0, 0), 0u);
    EXPECT_EQ(hist.windowTokenCount(), 50u);

    hist.resetWindow();

    EXPECT_EQ(hist.windowGeneration(), 1u);
    EXPECT_EQ(hist.windowTokenCount(), 0u);
    EXPECT_EQ(hist.activationCount(0, 0), 0u);
    EXPECT_EQ(hist.activationCount(0, 1), 0u);
    EXPECT_FLOAT_EQ(hist.weightedActivation(0, 0), 0.0f);
    EXPECT_FALSE(hist.windowFull());
}

TEST(Test__DecodeExpertHistogram, WeightedActivations)
{
    auto cfg = makeConfig(1, 4, 2, 256);
    DecodeExpertHistogram hist(cfg);

    // Token 1: expert 0 has weight 0.8, expert 1 has weight 0.2
    int idx1[] = {0, 1};
    float w1[] = {0.8f, 0.2f};
    hist.record(0, idx1, w1, 2);

    // Token 2: expert 0 has weight 0.3, expert 2 has weight 0.7
    int idx2[] = {0, 2};
    float w2[] = {0.3f, 0.7f};
    hist.record(0, idx2, w2, 2);

    EXPECT_NEAR(hist.weightedActivation(0, 0), 1.1f, 1e-5f);
    EXPECT_NEAR(hist.weightedActivation(0, 1), 0.2f, 1e-5f);
    EXPECT_NEAR(hist.weightedActivation(0, 2), 0.7f, 1e-5f);
    EXPECT_FLOAT_EQ(hist.weightedActivation(0, 3), 0.0f);
}

TEST(Test__DecodeExpertHistogram, TopExperts_SortedByCount)
{
    auto cfg = makeConfig(1, 8, 1, 256);
    DecodeExpertHistogram hist(cfg);

    // Record different counts per expert
    float w[] = {1.0f};
    for (int e = 0; e < 8; ++e)
    {
        int idx[] = {e};
        for (int i = 0; i < (e + 1) * 10; ++i)
            hist.record(0, idx, w, 1);
    }
    // Expert 7: 80, Expert 6: 70, ..., Expert 0: 10

    auto top3 = hist.topExperts(0, 3);
    ASSERT_EQ(top3.size(), 3u);
    EXPECT_EQ(top3[0].first, 7);
    EXPECT_EQ(top3[0].second, 80u);
    EXPECT_EQ(top3[1].first, 6);
    EXPECT_EQ(top3[1].second, 70u);
    EXPECT_EQ(top3[2].first, 5);
    EXPECT_EQ(top3[2].second, 60u);
}

TEST(Test__DecodeExpertHistogram, TopExperts_RequestMoreThanExist)
{
    auto cfg = makeConfig(1, 4, 1, 256);
    DecodeExpertHistogram hist(cfg);

    float w[] = {1.0f};
    int idx[] = {0};
    hist.record(0, idx, w, 1);

    // Request top 10 but only 4 experts exist
    auto top = hist.topExperts(0, 10);
    EXPECT_EQ(top.size(), 4u);
}

TEST(Test__DecodeExpertHistogram, UpdatePlacement_ChangesSocketMapping)
{
    auto cfg = makeConfig(1, 4, 1, 256, 2);
    // Default: expert 0→sock0, 1→sock1, 2→sock0, 3→sock1
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0};
    float w[] = {1.0f};
    for (int i = 0; i < 100; ++i)
        hist.record(0, idx, w, 1);

    // Before update: all load on socket 0
    auto loads_before = hist.socketLoads(0);
    EXPECT_EQ(loads_before[0], 100u);
    EXPECT_EQ(loads_before[1], 0u);

    // Move expert 0 to socket 1
    std::vector<int> new_placement = {1, 1, 0, 0};
    hist.updatePlacement(new_placement);

    // After update: same counts, but expert 0 is now on socket 1
    auto loads_after = hist.socketLoads(0);
    EXPECT_EQ(loads_after[0], 0u);
    EXPECT_EQ(loads_after[1], 100u);
}

TEST(Test__DecodeExpertHistogram, ThreadSafety_ConcurrentRecords)
{
    auto cfg = makeConfig(1, 16, 4, 100000);
    DecodeExpertHistogram hist(cfg);

    const int num_threads = 8;
    const int records_per_thread = 1000;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back([&hist, t, records_per_thread]()
                             {
            int indices[4];
            float weights[4] = {0.4f, 0.3f, 0.2f, 0.1f};
            for (int i = 0; i < records_per_thread; ++i) {
                // Each thread selects different experts based on thread id
                for (int k = 0; k < 4; ++k)
                    indices[k] = (t * 4 + k) % 16;
                hist.record(0, indices, weights, 4);
            } });
    }

    for (auto &t : threads)
        t.join();

    // Total window tokens = num_threads * records_per_thread
    EXPECT_EQ(hist.windowTokenCount(), num_threads * records_per_thread);

    // Verify total activations across all experts sums correctly
    // Each record has top_k=4 experts, so total activations = tokens * 4
    uint64_t total = 0;
    for (int e = 0; e < 16; ++e)
        total += hist.activationCount(0, e);
    EXPECT_EQ(total, static_cast<uint64_t>(num_threads * records_per_thread * 4));
}

TEST(Test__DecodeExpertHistogram, AllocationFree_HotPath)
{
    auto cfg = makeConfig(4, 64, 8, 100000);
    DecodeExpertHistogram hist(cfg);

    int indices[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    float weights[8] = {0.2f, 0.15f, 0.13f, 0.12f, 0.1f, 0.1f, 0.1f, 0.1f};

    // If record() allocated, 10000 calls in a tight loop would be slow
    // and potentially cause memory fragmentation. This test validates
    // by construction that the hot path is allocation-free.
    for (int i = 0; i < 10000; ++i)
        hist.record(i % 4, indices, weights, 8);

    // 10000 record() calls cycling through 4 layers: only last-layer (3)
    // increments the window counter, so 10000/4 = 2500 token counts.
    EXPECT_EQ(hist.windowTokenCount(), 2500u);
}

TEST(Test__DecodeExpertHistogram, LargeScale_256Experts_36Layers)
{
    // Qwen3.5-scale: 256 experts, 36 MoE layers, top_k=8
    auto cfg = makeConfig(36, 256, 8, 512, 2);
    DecodeExpertHistogram hist(cfg);

    // Simulate 512 tokens of decode routing
    int indices[8];
    float weights[8] = {0.18f, 0.16f, 0.14f, 0.12f, 0.1f, 0.1f, 0.1f, 0.1f};

    for (int token = 0; token < 512; ++token)
    {
        // Rotate expert selection
        for (int k = 0; k < 8; ++k)
            indices[k] = (token * 8 + k) % 256;

        for (int l = 0; l < 36; ++l)
            hist.record(l, indices, weights, 8);
    }

    EXPECT_TRUE(hist.windowFull());
    // Window counter increments once per decode token (on last MoE layer)
    EXPECT_EQ(hist.windowTokenCount(), 512u);

    // Verify per-layer consistency
    for (int l = 0; l < 36; ++l)
    {
        auto layer_hist = hist.layerHistogram(l);
        uint64_t layer_total = std::accumulate(layer_hist.begin(), layer_hist.end(), uint64_t{0});
        // Each token activates 8 experts, 512 tokens per layer
        EXPECT_EQ(layer_total, 512u * 8u);
    }

    // Socket loads should be roughly balanced (round-robin placement + uniform routing)
    for (int l = 0; l < 36; ++l)
    {
        auto loads = hist.socketLoads(l);
        ASSERT_EQ(loads.size(), 2u);
        // With uniform routing across 256 experts with round-robin,
        // each socket should get roughly half the activations
        uint64_t total_load = loads[0] + loads[1];
        EXPECT_GT(loads[0], total_load / 4); // At least 25%
        EXPECT_GT(loads[1], total_load / 4);
    }

    // Summary should not crash
    auto summary = hist.layerSummary(0);
    EXPECT_FALSE(summary.empty());
}

TEST(Test__DecodeExpertHistogram, AverageSocketImbalance)
{
    auto cfg = makeConfig(3, 4, 2, 256, 2);
    DecodeExpertHistogram hist(cfg);

    // Perfectly balanced routing across all layers
    int idx[] = {0, 1};
    float w[] = {0.5f, 0.5f};
    for (int l = 0; l < 3; ++l)
    {
        for (int i = 0; i < 100; ++i)
            hist.record(l, idx, w, 2);
    }

    EXPECT_FLOAT_EQ(hist.averageSocketImbalance(), 1.0f);
}

TEST(Test__DecodeExpertHistogram, LayerSummary_Format)
{
    auto cfg = makeConfig(1, 8, 2, 256, 2);
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0, 1};
    float w[] = {0.5f, 0.5f};
    for (int i = 0; i < 10; ++i)
        hist.record(0, idx, w, 2);

    std::string summary = hist.layerSummary(0);
    EXPECT_NE(summary.find("Layer 0"), std::string::npos);
    EXPECT_NE(summary.find("total_activations=20"), std::string::npos);
    EXPECT_NE(summary.find("socket_loads="), std::string::npos);
    EXPECT_NE(summary.find("imbalance="), std::string::npos);
}

TEST(Test__DecodeExpertHistogram, MultipleWindowResets)
{
    auto cfg = makeConfig(1, 4, 1, 5);
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0};
    float w[] = {1.0f};

    // Fill and reset multiple windows
    for (int gen = 0; gen < 3; ++gen)
    {
        EXPECT_EQ(hist.windowGeneration(), static_cast<uint64_t>(gen));
        for (int i = 0; i < 5; ++i)
            hist.record(0, idx, w, 1);
        EXPECT_TRUE(hist.windowFull());
        EXPECT_EQ(hist.activationCount(0, 0), 5u);
        hist.resetWindow();
        EXPECT_EQ(hist.activationCount(0, 0), 0u);
        EXPECT_EQ(hist.windowTokenCount(), 0u);
    }
    EXPECT_EQ(hist.windowGeneration(), 3u);
}

TEST(Test__DecodeExpertHistogram, LayerHistogram_ReturnsCorrectSize)
{
    auto cfg = makeConfig(1, 32, 4, 256);
    DecodeExpertHistogram hist(cfg);

    auto layer_hist = hist.layerHistogram(0);
    EXPECT_EQ(layer_hist.size(), 32u);
    for (auto count : layer_hist)
        EXPECT_EQ(count, 0u);
}

TEST(Test__DecodeExpertHistogram, MergeLayerCountsMatchesHostRecordCounts)
{
    auto cfg = makeConfig(3, 8, 2, 256);
    DecodeExpertHistogram recorded(cfg);
    DecodeExpertHistogram merged(cfg);

    int idx_a[] = {1, 3};
    float w_a[] = {0.7f, 0.3f};
    int idx_b[] = {3, 5};
    float w_b[] = {0.6f, 0.4f};

    for (int token = 0; token < 4; ++token)
    {
        for (int layer = 0; layer < 3; ++layer)
        {
            const int *idx = (token % 2 == 0) ? idx_a : idx_b;
            const float *weights = (token % 2 == 0) ? w_a : w_b;
            recorded.record(layer, idx, weights, 2);
        }
    }

    for (int layer = 0; layer < 3; ++layer)
    {
        uint64_t counts[8] = {};
        counts[1] = 2;
        counts[3] = 4;
        counts[5] = 2;
        merged.mergeLayerCounts(layer, counts, 8, /*count_window_tokens=*/layer == 2);
    }

    for (int layer = 0; layer < 3; ++layer)
        EXPECT_EQ(merged.layerHistogram(layer), recorded.layerHistogram(layer));
    EXPECT_EQ(merged.windowTokenCount(), recorded.windowTokenCount());
}

TEST(Test__DecodeExpertHistogram, RuntimeSyncCallbackMergesOnceAndCanAdvanceWindowSeparately)
{
    auto cfg = makeConfig(2, 4, 2, 3);
    DecodeExpertHistogram hist(cfg);

    uint64_t layer0_counts[4] = {2, 0, 1, 1};
    uint64_t layer1_counts[4] = {0, 2, 0, 2};
    bool pending = true;
    hist.registerRuntimeHistogramSync([&]()
                                      {
        if (!pending)
            return true;
        hist.mergeLayerCounts(0, layer0_counts, 4);
        hist.mergeLayerCounts(1, layer1_counts, 4);
        pending = false;
        return true; });

    hist.recordTokenBoundary(0);
    EXPECT_EQ(hist.windowTokenCount(), 0u);
    hist.recordTokenBoundary(1);
    EXPECT_EQ(hist.windowTokenCount(), 1u);

    EXPECT_TRUE(hist.syncRuntimeHistograms());
    EXPECT_EQ(hist.activationCount(0, 0), 2u);
    EXPECT_EQ(hist.activationCount(0, 2), 1u);
    EXPECT_EQ(hist.activationCount(1, 1), 2u);
    EXPECT_EQ(hist.activationCount(1, 3), 2u);
    EXPECT_EQ(hist.windowTokenCount(), 1u);

    EXPECT_TRUE(hist.syncRuntimeHistograms());
    EXPECT_EQ(hist.activationCount(0, 0), 2u);
    EXPECT_EQ(hist.activationCount(1, 1), 2u);
}

TEST(Test__DecodeExpertHistogram, PrefillChunkRouteMergeCountsOnlyRealRows)
{
    auto cfg = makeConfig(2, 8, 2, 8);
    DomainExpertHistogram hist(cfg);

    const std::vector<int> chunk0_routes = {
        0, 1,
        1, 2,
        2, 3,
        99, 99};
    RoutedExpertHistogramMerge chunk0;
    chunk0.source = ExpertHistogramSource::PrefillChunk;
    chunk0.layer_idx = 1;
    chunk0.real_token_count = 3;
    chunk0.bucket_token_count = 4;
    chunk0.top_k = 2;

    auto result = hist.mergeRoutedExpertRows(chunk0_routes.data(), chunk0);

    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.tokens_counted, 3u);
    EXPECT_EQ(result.activations_merged, 6u);
    EXPECT_EQ(hist.windowTokenCount(), 3u);
    EXPECT_EQ(hist.activationCount(1, 0), 1u);
    EXPECT_EQ(hist.activationCount(1, 1), 2u);
    EXPECT_EQ(hist.activationCount(1, 2), 2u);
    EXPECT_EQ(hist.activationCount(1, 3), 1u);
    EXPECT_EQ(hist.activationCount(1, 7), 0u);

    const std::vector<int> chunk1_routes = {
        4, 5,
        5, 6,
        99, 99,
        99, 99};
    RoutedExpertHistogramMerge chunk1 = chunk0;
    chunk1.real_token_count = 2;

    result = hist.mergeRoutedExpertRows(chunk1_routes.data(), chunk1);

    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.tokens_counted, 2u);
    EXPECT_EQ(result.activations_merged, 4u);
    EXPECT_EQ(hist.windowTokenCount(), 5u);
    EXPECT_EQ(hist.activationCount(1, 4), 1u);
    EXPECT_EQ(hist.activationCount(1, 5), 2u);
    EXPECT_EQ(hist.activationCount(1, 6), 1u);
    EXPECT_EQ(hist.activationCount(1, 7), 0u);
}

TEST(Test__DecodeExpertHistogram, PrefillChunkRouteMergeRejectsInvalidRealRowsBeforeMutation)
{
    auto cfg = makeConfig(1, 4, 2, 8);
    DecodeExpertHistogram hist(cfg);

    const std::vector<int> routes = {
        0, 1,
        2, 99,
        3, 3};
    RoutedExpertHistogramMerge merge;
    merge.source = ExpertHistogramSource::PrefillChunk;
    merge.layer_idx = 0;
    merge.real_token_count = 2;
    merge.bucket_token_count = 3;
    merge.top_k = 2;

    auto result = hist.mergeRoutedExpertRows(routes.data(), merge);

    EXPECT_FALSE(result);
    EXPECT_NE(result.error.find("expert id"), std::string::npos);
    EXPECT_EQ(hist.windowTokenCount(), 0u);
    for (int expert = 0; expert < cfg.num_experts; ++expert)
        EXPECT_EQ(hist.activationCount(0, expert), 0u);
}
