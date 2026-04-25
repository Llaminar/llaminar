/**
 * @file Test__MoERebalanceController.cpp
 * @brief Unit tests for MoERebalanceController
 */

#include <gtest/gtest.h>
#include "execution/moe/MoERebalanceController.h"
#include <vector>

using namespace llaminar2;

// ── Helpers ───────────────────────────────────────────

static MoERebalanceController::Config makeConfig(
    MoERebalanceMode mode,
    int num_experts = 8,
    int num_sockets = 2,
    int num_layers = 2,
    int top_k = 2,
    int window_size = 16)
{
    MoERebalanceController::Config cfg;
    cfg.mode = mode;
    cfg.num_layers = num_layers;
    cfg.num_experts = num_experts;
    cfg.top_k = top_k;
    cfg.window_size = window_size;

    for (int s = 0; s < num_sockets; ++s)
        cfg.sockets.push_back(DeviceId(DeviceType::CPU, s));

    // Round-robin expert-to-socket
    cfg.initial_expert_to_socket.resize(num_experts);
    for (int e = 0; e < num_experts; ++e)
        cfg.initial_expert_to_socket[e] = e % num_sockets;

    cfg.rebalance_config.imbalance_threshold = 1.3f;
    cfg.rebalance_config.max_swaps_per_layer = 4;
    cfg.rebalance_config.max_total_swaps = 16;
    cfg.rebalance_config.min_improvement_ratio = 0.05f;
    cfg.rebalance_config.layer_cooldown_generations = 0; // no cooldown for tests
    cfg.rebalance_config.min_window_activations = 1;

    return cfg;
}

/// Fill the histogram window with balanced routing across all experts
static void fillWindowBalanced(DecodeExpertHistogram& hist, int window_size,
                               int num_layers, int num_experts, int top_k)
{
    for (int t = 0; t < window_size; ++t)
    {
        for (int l = 0; l < num_layers; ++l)
        {
            std::vector<int> indices(top_k);
            std::vector<float> weights(top_k);
            for (int k = 0; k < top_k; ++k)
            {
                indices[k] = (t * top_k + k) % num_experts;
                weights[k] = 1.0f / static_cast<float>(top_k);
            }
            hist.record(l, indices.data(), weights.data(), top_k);
        }
    }
}

/// Fill the histogram window with heavily skewed routing (all to experts 0,1)
static void fillWindowSkewed(DecodeExpertHistogram& hist, int window_size,
                             int num_layers, int top_k)
{
    for (int t = 0; t < window_size; ++t)
    {
        for (int l = 0; l < num_layers; ++l)
        {
            // Always route to experts 0 and 1 (both on socket 0 in round-robin)
            std::vector<int> indices(top_k);
            std::vector<float> weights(top_k);
            for (int k = 0; k < top_k; ++k)
            {
                indices[k] = k; // experts 0, 1
                weights[k] = 1.0f / static_cast<float>(top_k);
            }
            hist.record(l, indices.data(), weights.data(), top_k);
        }
    }
}

// ── Tests ─────────────────────────────────────────────

TEST(Test__MoERebalanceController, Construction_OffMode)
{
    auto cfg = makeConfig(MoERebalanceMode::OFF);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.mode(), MoERebalanceMode::OFF);
    EXPECT_EQ(ctrl.histogram(), nullptr);
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_EQ(ctrl.totalRebalances(), 0);
    EXPECT_EQ(ctrl.totalSwaps(), 0);
}

TEST(Test__MoERebalanceController, Construction_ObserveMode)
{
    auto cfg = makeConfig(MoERebalanceMode::OBSERVE);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.mode(), MoERebalanceMode::OBSERVE);
    EXPECT_NE(ctrl.histogram(), nullptr);
    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, Construction_DynamicMode)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.mode(), MoERebalanceMode::DYNAMIC);
    EXPECT_NE(ctrl.histogram(), nullptr);
    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, ShouldRebalance_WindowNotFull)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Record fewer tokens than window size
    int indices[] = {0, 1};
    float weights[] = {0.5f, 0.5f};
    ctrl.histogram()->record(0, indices, weights, 2);

    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, ShouldRebalance_WindowFull)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);

    EXPECT_TRUE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, Rebalance_NoImbalance)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Fill with balanced routing
    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ASSERT_TRUE(ctrl.shouldRebalance());

    auto result = ctrl.rebalance();

    // With balanced routing, proposal should be empty (no beneficial swaps)
    EXPECT_TRUE(result.empty());
    // Window should be reset
    EXPECT_FALSE(ctrl.shouldRebalance());
}

TEST(Test__MoERebalanceController, Rebalance_WithImbalance)
{
    // Placement: experts 0-5 on socket 0, experts 6-7 on socket 1
    // Routing skewed to experts 0,1 (both on socket 0) → socket 0 is heavily overloaded
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Skewed routing → heavy imbalance between sockets
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    ASSERT_TRUE(ctrl.shouldRebalance());

    auto result = ctrl.rebalance();

    // With heavy imbalance, rebalancer should propose swaps and return new placement
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(static_cast<int>(result.size()), 8); // Full placement vector
}

TEST(Test__MoERebalanceController, Rebalance_UpdatesPlacement)
{
    // Experts 0-5 on socket 0, 6-7 on socket 1
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    auto initial = ctrl.currentPlacement();

    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    auto result = ctrl.rebalance();

    if (!result.empty())
    {
        // Pair swaps move a heavy expert to socket 1 and a light expert to socket 0.
        // At least one expert's socket assignment should differ from initial.
        bool placement_changed = false;
        for (int e = 0; e < 8; ++e)
        {
            if (ctrl.currentPlacement()[e] != initial[e])
                placement_changed = true;
        }
        EXPECT_TRUE(placement_changed);
    }
}

TEST(Test__MoERebalanceController, Rebalance_ResetsWindow)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ASSERT_TRUE(ctrl.shouldRebalance());

    ctrl.rebalance();

    // Window should be reset after rebalance
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_FALSE(ctrl.histogram()->windowFull());
}

TEST(Test__MoERebalanceController, Rebalance_CountsTotal)
{
    // Experts 0-5 on socket 0, 6-7 on socket 1 for forced imbalance
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.totalRebalances(), 0);
    EXPECT_EQ(ctrl.totalSwaps(), 0);

    // First rebalance cycle
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    auto result1 = ctrl.rebalance();

    if (!result1.empty())
    {
        EXPECT_EQ(ctrl.totalRebalances(), 1);
        EXPECT_GT(ctrl.totalSwaps(), 0);
        int swaps_after_first = ctrl.totalSwaps();

        // Second rebalance cycle
        fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
        auto result2 = ctrl.rebalance();

        if (!result2.empty())
        {
            EXPECT_EQ(ctrl.totalRebalances(), 2);
            EXPECT_GE(ctrl.totalSwaps(), swaps_after_first);
        }
    }
}

TEST(Test__MoERebalanceController, ObserveMode_NeverRebalances)
{
    // Experts 0-5 on socket 0, 6-7 on socket 1 for extreme imbalance
    auto cfg = makeConfig(MoERebalanceMode::OBSERVE, 8, 2, 2, 2, 16);
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 6) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Fill window
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);

    // Window is full but mode is OBSERVE -> never triggers
    EXPECT_FALSE(ctrl.shouldRebalance());
    EXPECT_EQ(ctrl.totalRebalances(), 0);
}

TEST(Test__MoERebalanceController, LogHistogramSummary_NoThrow)
{
    // OFF mode
    {
        auto cfg = makeConfig(MoERebalanceMode::OFF);
        MoERebalanceController ctrl(cfg);
        EXPECT_NO_THROW(ctrl.logHistogramSummary());
    }

    // OBSERVE mode with some data
    {
        auto cfg = makeConfig(MoERebalanceMode::OBSERVE, 8, 2, 2, 2, 16);
        MoERebalanceController ctrl(cfg);
        int indices[] = {0, 1};
        float weights[] = {0.5f, 0.5f};
        ctrl.histogram()->record(0, indices, weights, 2);
        EXPECT_NO_THROW(ctrl.logHistogramSummary());
    }

    // DYNAMIC mode with full window
    {
        auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
        MoERebalanceController ctrl(cfg);
        fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
        EXPECT_NO_THROW(ctrl.logHistogramSummary());
    }
}

// ── Helper for targeted expert routing ────────────────

/// Fill histogram with routing that always activates the given experts
static void fillWindowWithExperts(DecodeExpertHistogram& hist, int window_size,
                                  int num_layers, const std::vector<int>& active_experts)
{
    int top_k = static_cast<int>(active_experts.size());
    for (int t = 0; t < window_size; ++t)
    {
        for (int l = 0; l < num_layers; ++l)
        {
            std::vector<float> weights(top_k, 1.0f / top_k);
            hist.record(l, active_experts.data(), weights.data(), top_k);
        }
    }
}

// ── rebalanceLPT() Tests ──────────────────────────────

TEST(Test__MoERebalanceController, RebalanceLPT_BalancedInput_NearPerfectBalance)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();

    // Verify near-perfect balance: compute load per socket
    // Re-fill histogram to measure post-LPT imbalance
    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);

    const auto& placement = ctrl.currentPlacement();
    // Count experts per socket
    int count_s0 = 0, count_s1 = 0;
    for (int e = 0; e < 8; ++e)
    {
        if (placement[e] == 0) count_s0++;
        else count_s1++;
    }
    // With balanced input, LPT should assign 4 experts per socket
    EXPECT_EQ(count_s0, 4);
    EXPECT_EQ(count_s1, 4);
}

TEST(Test__MoERebalanceController, RebalanceLPT_SkewedInput_ImproveBalance)
{
    // Round-robin: experts 0,2,4,6→socket0, experts 1,3,5,7→socket1
    // Skewed routing to experts 0,1 → socket0 gets expert 0, socket1 gets expert 1
    // Both are equally hot, so initially balanced. Use contiguous partition instead.
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Route exclusively to experts 0,1 (both on socket 0)
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);

    ctrl.rebalanceLPT();

    // LPT should move one of {0,1} to socket 1 for better balance
    const auto& placement = ctrl.currentPlacement();
    int hot_on_s0 = 0;
    for (int e : {0, 1})
    {
        if (placement[e] == 0) hot_on_s0++;
    }
    // Expect at most 1 hot expert per socket (LPT splits them)
    EXPECT_LE(hot_on_s0, 1);
}

TEST(Test__MoERebalanceController, RebalanceLPT_UpdatesPlacement)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    auto initial = ctrl.currentPlacement();

    // Skewed to experts 0,1 (both on socket 0) → LPT should change placement
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    ctrl.rebalanceLPT();

    const auto& updated = ctrl.currentPlacement();
    ASSERT_EQ(static_cast<int>(updated.size()), 8);

    // At least one expert should have moved
    bool any_changed = false;
    for (int e = 0; e < 8; ++e)
    {
        if (updated[e] != initial[e])
            any_changed = true;
    }
    EXPECT_TRUE(any_changed);
}

TEST(Test__MoERebalanceController, RebalanceLPT_ResetsWindow)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ASSERT_TRUE(ctrl.histogram()->windowFull());

    ctrl.rebalanceLPT();

    EXPECT_FALSE(ctrl.histogram()->windowFull());
}

TEST(Test__MoERebalanceController, RebalanceLPT_IncrementsRebalanceCount)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    EXPECT_EQ(ctrl.totalRebalances(), 0);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();
    EXPECT_EQ(ctrl.totalRebalances(), 1);

    // Second cycle
    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();
    EXPECT_EQ(ctrl.totalRebalances(), 2);
}

TEST(Test__MoERebalanceController, RebalanceLPT_ContiguousPartition_SkewedRouting)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Route only to experts 0-3 (all on socket 0)
    fillWindowWithExperts(*ctrl.histogram(), 16, 2, {0, 1, 2, 3});

    ctrl.rebalanceLPT();

    const auto& placement = ctrl.currentPlacement();
    // LPT should distribute experts 0-3 across both sockets
    int hot_on_s0 = 0, hot_on_s1 = 0;
    for (int e = 0; e < 4; ++e)
    {
        if (placement[e] == 0) hot_on_s0++;
        else hot_on_s1++;
    }
    // At least one of experts 0-3 should now be on socket 1
    EXPECT_GT(hot_on_s1, 0);
    // Should be roughly balanced: 2 hot experts per socket
    EXPECT_EQ(hot_on_s0, 2);
    EXPECT_EQ(hot_on_s1, 2);
}

TEST(Test__MoERebalanceController, RebalanceLPT_MultiSocket)
{
    // 4 sockets, 16 experts
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, /*num_experts=*/16, /*num_sockets=*/4,
                          /*num_layers=*/2, /*top_k=*/4, /*window_size=*/16);
    MoERebalanceController ctrl(cfg);

    // Route to experts 0-3 (all on socket 0 in round-robin with 4 sockets)
    fillWindowWithExperts(*ctrl.histogram(), 16, 2, {0, 1, 2, 3});

    ctrl.rebalanceLPT();

    const auto& placement = ctrl.currentPlacement();
    // LPT should spread experts 0-3 across all 4 sockets
    std::vector<int> socket_for_hot(4);
    for (int e = 0; e < 4; ++e)
        socket_for_hot[e] = placement[e];

    // With 4 equally-loaded hot experts and 4 sockets, each should go to a different socket
    std::sort(socket_for_hot.begin(), socket_for_hot.end());
    // All 4 hot experts on distinct sockets
    EXPECT_EQ(socket_for_hot[0], 0);
    EXPECT_EQ(socket_for_hot[1], 1);
    EXPECT_EQ(socket_for_hot[2], 2);
    EXPECT_EQ(socket_for_hot[3], 3);
}

TEST(Test__MoERebalanceController, RebalanceLPT_ZeroCountExperts)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Only experts 0,1 are active — experts 2-7 have zero counts
    fillWindowWithExperts(*ctrl.histogram(), 16, 2, {0, 1});

    ctrl.rebalanceLPT();

    const auto& placement = ctrl.currentPlacement();

    // All 8 experts should still have valid placements (0 or 1)
    for (int e = 0; e < 8; ++e)
    {
        EXPECT_GE(placement[e], 0);
        EXPECT_LE(placement[e], 1);
    }

    // LPT balances by LOAD, not expert count.
    // The 2 hot experts should be split across sockets for load balance.
    int hot_on_s0 = 0, hot_on_s1 = 0;
    for (int e : {0, 1})
    {
        if (placement[e] == 0) hot_on_s0++;
        else hot_on_s1++;
    }
    EXPECT_EQ(hot_on_s0, 1);
    EXPECT_EQ(hot_on_s1, 1);

    // Zero-count experts all tie at load=0, so LPT tie-breaks to lowest-index socket.
    // They don't affect balance — just verify they're all placed somewhere valid.
    for (int e = 2; e < 8; ++e)
    {
        EXPECT_GE(placement[e], 0);
        EXPECT_LE(placement[e], 1);
    }
}

// ── computeExpertMasks() Tests ────────────────────────

TEST(Test__MoERebalanceController, ComputeExpertMasks_InitialPartition)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    // Round-robin: experts 0,2,4,6→socket0; experts 1,3,5,7→socket1
    auto masks_s0 = ctrl.computeExpertMasks(0);
    auto masks_s1 = ctrl.computeExpertMasks(1);

    ASSERT_EQ(static_cast<int>(masks_s0.size()), 2); // num_layers
    ASSERT_EQ(static_cast<int>(masks_s0[0].size()), 8); // num_experts

    // Check socket 0 mask matches round-robin
    for (int l = 0; l < 2; ++l)
    {
        EXPECT_TRUE(masks_s0[l][0]);   // expert 0 → socket 0
        EXPECT_FALSE(masks_s0[l][1]);  // expert 1 → socket 1
        EXPECT_TRUE(masks_s0[l][2]);   // expert 2 → socket 0
        EXPECT_FALSE(masks_s0[l][3]);  // expert 3 → socket 1
        EXPECT_TRUE(masks_s0[l][4]);   // expert 4 → socket 0
        EXPECT_FALSE(masks_s0[l][5]);  // expert 5 → socket 1
        EXPECT_TRUE(masks_s0[l][6]);   // expert 6 → socket 0
        EXPECT_FALSE(masks_s0[l][7]);  // expert 7 → socket 1
    }

    // Check socket 1 is the complement
    for (int l = 0; l < 2; ++l)
    {
        EXPECT_FALSE(masks_s1[l][0]);
        EXPECT_TRUE(masks_s1[l][1]);
        EXPECT_FALSE(masks_s1[l][2]);
        EXPECT_TRUE(masks_s1[l][3]);
        EXPECT_FALSE(masks_s1[l][4]);
        EXPECT_TRUE(masks_s1[l][5]);
        EXPECT_FALSE(masks_s1[l][6]);
        EXPECT_TRUE(masks_s1[l][7]);
    }
}

TEST(Test__MoERebalanceController, ComputeExpertMasks_AfterLPT)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    // Contiguous: 0-3→socket0, 4-7→socket1
    for (int e = 0; e < 8; ++e)
        cfg.initial_expert_to_socket[e] = (e < 4) ? 0 : 1;
    MoERebalanceController ctrl(cfg);

    // Skewed to experts 0,1 → LPT will move one to socket 1
    fillWindowSkewed(*ctrl.histogram(), 16, 2, 2);
    ctrl.rebalanceLPT();

    const auto& placement = ctrl.currentPlacement();
    auto masks_s0 = ctrl.computeExpertMasks(0);

    // Masks should reflect the updated LPT placement
    for (int l = 0; l < 2; ++l)
    {
        for (int e = 0; e < 8; ++e)
        {
            EXPECT_EQ(masks_s0[l][e], placement[e] == 0)
                << "Mismatch at layer=" << l << " expert=" << e;
        }
    }
}

TEST(Test__MoERebalanceController, ComputeExpertMasks_SocketComplementary)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, 2, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 2, 8, 2);
    ctrl.rebalanceLPT();

    auto masks_s0 = ctrl.computeExpertMasks(0);
    auto masks_s1 = ctrl.computeExpertMasks(1);

    for (int l = 0; l < 2; ++l)
    {
        for (int e = 0; e < 8; ++e)
        {
            // Exactly one socket owns each expert (no overlap, no gap)
            EXPECT_NE(masks_s0[l][e], masks_s1[l][e])
                << "Overlap or gap at layer=" << l << " expert=" << e;
        }
    }
}

TEST(Test__MoERebalanceController, ComputeExpertMasks_AllLayersSameGlobalPartition)
{
    auto cfg = makeConfig(MoERebalanceMode::DYNAMIC, 8, 2, /*num_layers=*/4, 2, 16);
    MoERebalanceController ctrl(cfg);

    fillWindowBalanced(*ctrl.histogram(), 16, 4, 8, 2);
    ctrl.rebalanceLPT();

    auto masks = ctrl.computeExpertMasks(0);
    ASSERT_EQ(static_cast<int>(masks.size()), 4);

    // All layers should have identical masks (global LPT, not per-layer)
    for (int l = 1; l < 4; ++l)
    {
        EXPECT_EQ(masks[l], masks[0])
            << "Layer " << l << " mask differs from layer 0";
    }
}
