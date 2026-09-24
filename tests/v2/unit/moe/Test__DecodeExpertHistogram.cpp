/**
 * @file Test__DecodeExpertHistogram.cpp
 * @brief Device-free proofs of routing demand, bank retirement and physical ownership.
 *
 * Transaction tests exercise the actual histogram ingress and RCU rotation, not
 * a parallel recorder. Counts and complete batches must survive as one immutable
 * sample while inference continues into the next preallocated generation.
 */

#include <gtest/gtest.h>
#include "execution/moe/DecodeExpertHistogram.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <thread>
#include <vector>

using namespace llaminar2;

/**
 * @brief Build a complete ownership table by repeating one static owner row.
 *
 * Static ordinal and random placement intentionally start with the same owner
 * row in every routed layer. Dynamic movement is allowed to diverge from that
 * initial table layer by layer after histogram evidence is available.
 */
static MoELayeredExpertOwnership makeOwnership(
    int num_layers,
    int num_sockets,
    const std::vector<int> &owners)
{
    return MoELayeredExpertOwnership::uniform(
        num_layers, num_sockets, owners);
}

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

    // Default: repeat one round-robin owner row in every routed layer.
    std::vector<int> owners(static_cast<size_t>(num_experts));
    for (int e = 0; e < num_experts; ++e)
        owners[static_cast<size_t>(e)] = e % num_sockets;
    cfg.ownership = makeOwnership(num_layers, num_sockets, owners);

    return cfg;
}

/** @return A CPU physical ledger admitting exactly the requested test payload. */
static std::shared_ptr<PhysicalMemoryAuthority> transactionMemory(size_t bytes)
{
    PhysicalMemoryBOMBuilder bom({.world_rank = 0, .device = DeviceId::cpu(),
                                  .total_bytes = bytes, .admission_available_bytes = bytes});
    bom.add(PhysicalMemoryOwner::ExecutionWorkspace, bytes);
    PhysicalMemoryPlanBuilder plan;
    plan.add(bom.build());
    return std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
}

/** @return Current physical payload claims, including still-retained snapshots. */
static size_t transactionClaims(const std::shared_ptr<PhysicalMemoryAuthority> &memory)
{
    return memory->claimedBytes(DeviceId::cpu(), PhysicalMemoryOwner::ExecutionWorkspace,
                                PhysicalMemoryMaterializationKind::NewAllocation);
}

TEST(Test__DecodeExpertHistogram, TransactionSampleClosesCountsAtWholeBatchBoundary)
{
    auto cfg = makeConfig(1, 4, 2, 3);
    const moe_overlay_economy::TransactionDemandCapacity capacity{3, 4, 2};
    auto memory = transactionMemory(8192);
    cfg.transaction_demand = ExpertHistogramTransactionConfig{capacity, memory};
    DecodeExpertHistogram hist(cfg);
    const int rows[]{0, 1, 2, 3, 0, 3, 1, 2, -1, -1};
    std::array<uint64_t, 4> scratch{};
    const RoutedExpertHistogramMerge batch{.source = ExpertHistogramSource::PrefillChunk,
        .layer_idx = 0, .real_token_count = 4, .bucket_token_count = 5,
        .top_k = 2, .route_stride = 2, .count_window_tokens = true};
    ASSERT_TRUE(hist.mergeRoutedExpertRows(rows, batch, scratch));
    const auto closed = hist.mergeRoutedExpertRows(rows, batch, scratch);
    ASSERT_TRUE(closed);
    EXPECT_EQ(closed.activations_merged, 0u);
    EXPECT_EQ(closed.tokens_counted, 0u);
    auto window = hist.freezeAndRotateWindow();
    ASSERT_TRUE(window.valid());
    ASSERT_NE(window.transaction_demand, nullptr);
    EXPECT_EQ(&window.validatedView().transactionDemand(), window.transaction_demand.get());
    EXPECT_EQ(window.token_count, 4u);
    EXPECT_EQ(window.expert_counts, (std::vector<uint64_t>{2, 2, 2, 2}));
    const auto transactions = window.transaction_demand->layerTransactions(0);
    ASSERT_EQ(transactions.size(), 1u);
    EXPECT_EQ(transactions[0].phase, ExpertHistogramSource::PrefillChunk);
    EXPECT_EQ(transactions[0].logical_rows, 4u);
    const auto retained = window.transaction_demand->routes(0, 0);
    EXPECT_EQ(retained.logical_rows, 4u);
    EXPECT_EQ(retained.capacity_slots, 8u);
    EXPECT_TRUE(std::equal(rows, rows + 8, retained.expert_ids));
    // A rounded forecast cannot masquerade as this observed transaction sample.
    ++window.expert_counts[0];
    ++window.source_expert_counts[4];
    EXPECT_FALSE(window.valid());
}

TEST(Test__DecodeExpertHistogram, TransactionSnapshotsOutliveBankReuseAndHistogram)
{
    auto memory = transactionMemory(8192);
    DecodeExpertHistogramWindow retained;
    {
        auto cfg = makeConfig(1, 4, 2, 2);
        const moe_overlay_economy::TransactionDemandCapacity capacity{2, 2, 2};
        cfg.transaction_demand = ExpertHistogramTransactionConfig{capacity, memory};
        DecodeExpertHistogram hist(cfg);
        const auto banks = 2 * capacity.allocationBytes();
        EXPECT_EQ(transactionClaims(memory), banks);
        std::array<uint64_t, 4> scratch{};
        int ids[]{0, 1};
        const RoutedExpertHistogramMerge row{.source = ExpertHistogramSource::DecodeToken,
            .layer_idx = 0, .real_token_count = 1, .bucket_token_count = 1,
            .top_k = 2, .route_stride = 2, .count_window_tokens = true};
        ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, row, scratch));
        retained = hist.freezeAndRotateWindow();
        ASSERT_NE(retained.transaction_demand, nullptr);
        EXPECT_EQ(transactionClaims(memory), banks + retained.transaction_demand->allocationBytes());
        ids[0] = 2;
        ids[1] = 3;
        for (int epoch = 0; epoch < 20; ++epoch)
        {
            ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, row, scratch));
            auto next = hist.freezeAndRotateWindow();
            EXPECT_TRUE(next.valid());
            EXPECT_EQ(next.transaction_demand->routes(0, 0).expert_ids[0], 2);
            EXPECT_EQ(retained.transaction_demand->routes(0, 0).expert_ids[0], 0);
            EXPECT_TRUE(retained.valid());
        }
    }
    EXPECT_EQ(transactionClaims(memory), retained.transaction_demand->allocationBytes());
    EXPECT_TRUE(retained.valid());
    retained = {};
    EXPECT_EQ(transactionClaims(memory), 0u);
}

TEST(Test__DecodeExpertHistogram, TransactionAdmissionRejectsPartialAndUnownedEvidence)
{
    auto cfg = makeConfig(1, 4, 2, 2);
    const moe_overlay_economy::TransactionDemandCapacity capacity{2, 4, 2};
    cfg.transaction_demand = ExpertHistogramTransactionConfig{capacity, nullptr};
    EXPECT_THROW(DecodeExpertHistogram{cfg}, std::invalid_argument);
    auto insufficient = transactionMemory(capacity.allocationBytes());
    cfg.transaction_demand->memory = insufficient;
    EXPECT_THROW(DecodeExpertHistogram{cfg}, std::logic_error);
    EXPECT_EQ(transactionClaims(insufficient), 0u);
    cfg.transaction_demand->memory = transactionMemory(8192);
    DecodeExpertHistogram hist(cfg);
    EXPECT_THROW(hist.setWindowSize(3), std::invalid_argument);
    EXPECT_EQ(hist.windowSize(), 2);
    const uint64_t counts[]{1, 1, 0, 0};
    EXPECT_THROW(hist.mergeLayerCounts(0, counts, 4, true), std::logic_error);
    EXPECT_THROW(hist.recordTokenBoundary(0), std::logic_error);
    int ids[]{0, 1, 2, -1};
    std::array<uint64_t, 4> scratch{};
    const RoutedExpertHistogramMerge batch{.source = ExpertHistogramSource::PrefillChunk,
        .layer_idx = 0, .real_token_count = 2, .bucket_token_count = 2,
        .top_k = 2, .route_stride = 2, .count_window_tokens = true};
    EXPECT_FALSE(hist.mergeRoutedExpertRows(ids, batch, scratch));
    const auto frozen = hist.freezeAndRotateWindow();
    EXPECT_TRUE(frozen.valid());
    EXPECT_EQ(frozen.token_count, 0u);
    EXPECT_TRUE(frozen.transaction_demand->layerTransactions(0).empty());
}

TEST(Test__DecodeExpertHistogram, TransactionPhaseAndAdaptiveBudgetUseCanonicalParentState)
{
    auto cfg = makeConfig(2, 4, 2, 1);
    cfg.token_boundary_layer_idx = 0; // Retained auxiliary layers need not advance main-model tokens.
    cfg.transaction_demand = ExpertHistogramTransactionConfig{{4, 4, 2}, transactionMemory(8192)};
    DecodeExpertHistogram hist(cfg);
    const int ids[]{0, 1, 2, 3};
    const float weights[]{0.5f, 0.5f};
    hist.record(0, ids, weights, 2);
    hist.record(0, ids, weights, 2); // Parent target is one: neither representation grows.
    hist.setWindowSize(4);
    std::array<uint64_t, 4> scratch{};
    const RoutedExpertHistogramMerge grouped{.source = ExpertHistogramSource::GroupedVerifier,
        .layer_idx = 0, .real_token_count = 2, .bucket_token_count = 2,
        .top_k = 2, .route_stride = 2, .count_window_tokens = true};
    ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, grouped, scratch));
    auto auxiliary = grouped;
    auxiliary.layer_idx = 1;
    ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, auxiliary, scratch));
    auto prefill = grouped;
    prefill.real_token_count = 1;
    prefill.source = ExpertHistogramSource::PrefillChunk;
    ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, prefill, scratch));
    const auto frozen = hist.freezeAndRotateWindow();
    ASSERT_TRUE(frozen.valid());
    EXPECT_EQ(frozen.token_count, 4u);
    EXPECT_EQ(frozen.source_token_counts, (std::array<uint64_t, 3>{1, 1, 2}));
    const auto batches = frozen.transaction_demand->layerTransactions(0);
    ASSERT_EQ(batches.size(), 3u);
    EXPECT_EQ(batches[0].phase, ExpertHistogramSource::DecodeToken);
    EXPECT_EQ(batches[1].phase, ExpertHistogramSource::GroupedVerifier);
    EXPECT_EQ(batches[1].logical_rows, 2u);
    EXPECT_EQ(batches[2].phase, ExpertHistogramSource::PrefillChunk);
    EXPECT_EQ(frozen.transaction_demand->layerTransactions(1).size(), 1u);
    EXPECT_THROW((void)frozen.transaction_demand->routes(-1, 0), std::out_of_range);
    EXPECT_THROW((void)frozen.transaction_demand->routes(0, 3), std::out_of_range);
    EXPECT_THROW((void)frozen.transaction_demand->layerTransactions(2), std::out_of_range);
}

TEST(Test__DecodeExpertHistogram, TransactionQuarantineUsesExistingAdmissionAndRotation)
{
    auto cfg = makeConfig(1, 4, 2, 4);
    cfg.transaction_demand = ExpertHistogramTransactionConfig{{4, 2, 2}, transactionMemory(8192)};
    DecodeExpertHistogram hist(cfg);
    int ids[]{0, 1};
    std::array<uint64_t, 4> scratch{};
    const RoutedExpertHistogramMerge row{.source = ExpertHistogramSource::DecodeToken,
        .layer_idx = 0, .real_token_count = 1, .bucket_token_count = 1,
        .top_k = 2, .route_stride = 2, .count_window_tokens = true};
    ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, row, scratch));
    hist.beginOptimizationDemandRebase();
    auto ignored = hist.mergeRoutedExpertRows(ids, row, scratch);
    ASSERT_TRUE(ignored);
    EXPECT_EQ(ignored.activations_merged, 0u);
    const auto calibration = hist.freezeAndRotateWindow();
    ASSERT_TRUE(calibration.valid());
    EXPECT_EQ(calibration.token_count, 1u);
    ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, row, scratch));
    hist.activateOptimizationDemand();
    ids[0] = 2;
    ids[1] = 3;
    ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, row, scratch));
    const auto optimization = hist.freezeAndRotateWindow();
    ASSERT_TRUE(optimization.valid());
    EXPECT_EQ(optimization.token_count, 1u);
    EXPECT_EQ(optimization.expert_counts, (std::vector<uint64_t>{0, 0, 1, 1}));
    EXPECT_EQ(optimization.transaction_demand->routes(0, 0).expert_ids[0], 2);
}

TEST(Test__DecodeExpertHistogram, TransactionConcurrentRotationRetainsEveryAdmittedBatch)
{
    auto cfg = makeConfig(1, 4, 2, 512);
    cfg.transaction_demand = ExpertHistogramTransactionConfig{{512, 2, 2}, transactionMemory(1u << 20)};
    DecodeExpertHistogram hist(cfg);
    std::atomic<bool> done{false};
    uint64_t admitted = 0;
    std::thread producer([&] {
        const int ids[]{0, 1};
        std::array<uint64_t, 4> scratch{};
        const RoutedExpertHistogramMerge row{.source = ExpertHistogramSource::DecodeToken,
            .layer_idx = 0, .real_token_count = 1, .bucket_token_count = 1,
            .top_k = 2, .route_stride = 2, .count_window_tokens = true};
        for (int token = 0; token < 4096; ++token)
        {
            auto result = hist.mergeRoutedExpertRows(ids, row, scratch);
            EXPECT_TRUE(result);
            admitted += result.tokens_counted;
            if (token % 32 == 0) std::this_thread::yield();
        }
        done.store(true, std::memory_order_release);
    });
    uint64_t observed = 0;
    do
    {
        const auto frozen = hist.freezeAndRotateWindow();
        EXPECT_TRUE(frozen.valid());
        EXPECT_EQ(frozen.transaction_demand->layerTransactions(0).size(), frozen.token_count);
        observed += frozen.token_count;
    } while (!done.load(std::memory_order_acquire));
    producer.join();
    const auto tail = hist.freezeAndRotateWindow();
    EXPECT_TRUE(tail.valid());
    observed += tail.token_count;
    EXPECT_GT(admitted, 0u);
    EXPECT_EQ(observed, admitted);
}

TEST(Test__DecodeExpertHistogram, TransactionBOMCoversExactWorstCaseWithoutReserve)
{
    using namespace moe_overlay_economy;
    const TransactionDemandCapacity capacity{3, 4, 2};
    const auto snapshot_bytes = DecodeExpertTransactionWindow::maximumAllocationBytes(capacity, 1, 4);
    const auto bank_bytes = 2 * capacity.allocationBytes();
    auto memory = transactionMemory(bank_bytes + snapshot_bytes);
    auto cfg = makeConfig(1, 4, 2, 3);
    cfg.transaction_demand = ExpertHistogramTransactionConfig{capacity, memory};
    DecodeExpertHistogram hist(cfg);
    int ids[]{0, 1, 2, 3, 0, 2, 1, 3};
    float weights[]{0.5f, 0.5f};
    hist.record(0, ids, weights, 2);
    hist.record(0, ids, weights, 2);
    std::array<uint64_t, 4> scratch{};
    const RoutedExpertHistogramMerge batch{.source = ExpertHistogramSource::PrefillChunk,
        .layer_idx = 0, .real_token_count = 4, .bucket_token_count = 4,
        .top_k = 2, .route_stride = 2, .count_window_tokens = true};
    ASSERT_TRUE(hist.mergeRoutedExpertRows(ids, batch, scratch));
    EXPECT_EQ(transactionClaims(memory), bank_bytes);
    const auto frozen = hist.freezeAndRotateWindow();
    ASSERT_TRUE(frozen.valid());
    EXPECT_EQ(frozen.token_count, 6u);
    EXPECT_EQ(frozen.transaction_demand->allocationBytes(), snapshot_bytes);
    EXPECT_EQ(transactionClaims(memory), bank_bytes + snapshot_bytes);
    EXPECT_THROW((void)DecodeExpertTransactionWindow::maximumAllocationBytes({}, 1, 4), std::invalid_argument);
    EXPECT_THROW((void)DecodeExpertTransactionWindow::maximumAllocationBytes(capacity, 0, 4), std::invalid_argument);
}

TEST(Test__DecodeExpertHistogram, TransactionIngressPreservesOppositePayoffsWithIdenticalMarginals)
{
    using namespace moe_overlay_economy;
    auto cfg = makeConfig(1, 4, 2, 2);
    cfg.transaction_demand = ExpertHistogramTransactionConfig{{2, 1, 2}, transactionMemory(8192)};
    DecodeExpertHistogram hist(cfg);
    const float weights[]{0.5f, 0.5f};
    const int grouped_routes[]{0, 1, 2, 3};
    const int mixed_routes[]{0, 3, 1, 2};
    hist.record(0, grouped_routes, weights, 2);
    hist.record(0, grouped_routes + 2, weights, 2);
    const auto grouped = hist.freezeAndRotateWindow();
    hist.record(0, mixed_routes, weights, 2);
    hist.record(0, mixed_routes + 2, weights, 2);
    const auto mixed = hist.freezeAndRotateWindow();
    ASSERT_TRUE(grouped.valid());
    ASSERT_TRUE(mixed.valid());
    EXPECT_EQ(grouped.source_expert_counts, mixed.source_expert_counts);
    const int32_t before[]{0, 0, 1, 1};
    const int32_t after[]{1, 0, 0, 1};
    const uint64_t prices[]{1, 3};
    const TransactionPlacementCosts placement{before, after, prices, 4, 2};
    const auto score = [&](const DecodeExpertHistogramWindow &window) {
        ServiceCostPair sum;
        std::array<ServiceCostPair, 2> scratch{};
        for (size_t batch = 0; batch < window.transaction_demand->layerTransactions(0).size(); ++batch)
        {
            const auto cost = scoreTransaction(window.transaction_demand->routes(0, batch),
                                               placement, scratch.data(), scratch.size());
            EXPECT_EQ(appendTransaction(sum, cost), TransactionCostStatus::Complete);
        }
        return sum;
    };
    EXPECT_EQ(score(grouped).before_ns, 8u);
    EXPECT_EQ(score(grouped).after_ns, 6u);
    EXPECT_EQ(score(mixed).before_ns, 6u);
    EXPECT_EQ(score(mixed).after_ns, 8u);
}

// ── Tests ─────────────────────────────────────────────

TEST(Test__DecodeExpertHistogram,
     RetainedMTPTopologyKeepsMainDecodeCatchupReachable)
{
    const auto topology =
        ExpertHistogramProductionTopology::forRetainedExecution(
            /*retained_layer_count=*/49,
            /*main_inference_layer_count=*/48,
            ExpertHistogramServingRegime::PositiveDepthMTP);

    EXPECT_EQ(topology.layerCount(), 49u);
    EXPECT_TRUE(topology.reachable(0, 0));
    EXPECT_TRUE(topology.reachable(0, 1));
    EXPECT_TRUE(topology.reachable(0, 2));
    EXPECT_TRUE(topology.reachable(47, 0));
    EXPECT_FALSE(topology.requiresServiceEvidence(0, 0));
    EXPECT_TRUE(topology.requiresServiceEvidence(0, 1));
    EXPECT_TRUE(topology.requiresServiceEvidence(0, 2));
    EXPECT_FALSE(topology.reachable(48, 0));
    EXPECT_FALSE(topology.reachable(48, 1));
    EXPECT_TRUE(topology.reachable(48, 2));
    EXPECT_EQ(
        topology.activeSources(),
        (ExpertHistogramProductionSourceMask{true, true, true}));
    EXPECT_EQ(
        topology.economyActiveSources(),
        (ExpertHistogramProductionSourceMask{false, true, true}));
}

TEST(Test__DecodeExpertHistogram,
     DisabledMTPLeavesRetainedPredictorCapacityPhaseEmpty)
{
    const auto topology =
        ExpertHistogramProductionTopology::forRetainedExecution(
            /*retained_layer_count=*/49,
            /*main_inference_layer_count=*/48,
            ExpertHistogramServingRegime::Serial);

    EXPECT_TRUE(topology.reachable(0, 0));
    EXPECT_TRUE(topology.reachable(0, 1));
    EXPECT_FALSE(topology.reachable(0, 2));
    EXPECT_FALSE(topology.reachable(48, 0));
    EXPECT_FALSE(topology.reachable(48, 1));
    EXPECT_FALSE(topology.reachable(48, 2));
    EXPECT_TRUE(topology.requiresServiceEvidence(0, 0));
    EXPECT_TRUE(topology.requiresServiceEvidence(0, 1));
    EXPECT_EQ(
        topology.activeSources(),
        (ExpertHistogramProductionSourceMask{true, true, false}));
}

TEST(Test__DecodeExpertHistogram,
     AdaptiveDepthZeroMakesSerialDecodeEconomyRequired)
{
    const auto topology =
        ExpertHistogramProductionTopology::forRetainedExecution(
            /*retained_layer_count=*/49,
            /*main_inference_layer_count=*/48,
            ExpertHistogramServingRegime::AdaptiveSerialOrMTP);

    EXPECT_TRUE(topology.reachable(0, 0));
    EXPECT_TRUE(topology.requiresServiceEvidence(0, 0));
    EXPECT_EQ(
        topology.economyActiveSources(),
        (ExpertHistogramProductionSourceMask{true, true, true}));
    EXPECT_FALSE(topology.requiresServiceEvidence(48, 0));
    EXPECT_TRUE(topology.requiresServiceEvidence(48, 2));
}

TEST(Test__DecodeExpertHistogram,
     RetainedExecutionTopologyRejectsInvalidLayerBoundaries)
{
    EXPECT_THROW(
        (void)ExpertHistogramProductionTopology::forRetainedExecution(
            0, 0, ExpertHistogramServingRegime::PositiveDepthMTP),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ExpertHistogramProductionTopology::forRetainedExecution(
            48, 0, ExpertHistogramServingRegime::PositiveDepthMTP),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ExpertHistogramProductionTopology::forRetainedExecution(
            48, 49, ExpertHistogramServingRegime::PositiveDepthMTP),
        std::invalid_argument);
}

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

TEST(Test__DecodeExpertHistogram,
     AdaptiveWindowPublicationIsRaceSafeAndChangesDemandCapacity)
{
    DecodeExpertHistogram hist(makeConfig(1, 8, 2, 4));
    EXPECT_EQ(hist.windowSize(), 4);

    hist.recordTokenBoundary(0, 4u);
    EXPECT_TRUE(hist.windowFull());
    (void)hist.freezeAndRotateWindow();
    hist.setWindowSize(16);

    EXPECT_EQ(hist.windowSize(), 16);
    const auto demand = hist.optimizationDemandWindow();
    EXPECT_EQ(demand.generation, 1u);
    EXPECT_EQ(demand.capacity_routed_rows, 16u);
    EXPECT_FALSE(hist.windowFull());
    EXPECT_THROW(hist.setWindowSize(0), std::invalid_argument);
}

TEST(Test__DecodeExpertHistogram, OptimizationDemandWindowTracksActiveRCUBank)
{
    auto cfg = makeConfig(1, 8, 2, 4);
    DecodeExpertHistogram hist(cfg);

    const auto initial = hist.optimizationDemandWindow();
    EXPECT_TRUE(initial.valid());
    EXPECT_EQ(initial.generation, 0u);
    EXPECT_EQ(initial.collected_routed_rows, 0u);
    EXPECT_EQ(initial.capacity_routed_rows, 4u);
    EXPECT_EQ(initial.remainingRoutedRows(), 4u);
    EXPECT_TRUE(initial.canAdmitExclusiveCohort(3u));
    EXPECT_FALSE(initial.canAdmitExclusiveCohort(4u));

    hist.recordTokenBoundary(0, 4u);
    const auto complete = hist.optimizationDemandWindow();
    EXPECT_EQ(complete.generation, 0u);
    EXPECT_EQ(complete.collected_routed_rows, 4u);
    EXPECT_EQ(complete.remainingRoutedRows(), 0u);
    EXPECT_FALSE(complete.canAdmitExclusiveCohort(1u));

    const auto frozen = hist.freezeAndRotateWindow();
    EXPECT_EQ(frozen.generation, 0u);
    EXPECT_EQ(frozen.token_count, 4u);
    const auto rotated = hist.optimizationDemandWindow();
    EXPECT_EQ(rotated.generation, 1u);
    EXPECT_EQ(rotated.collected_routed_rows, 0u);
    EXPECT_EQ(rotated.capacity_routed_rows, 4u);
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

TEST(Test__DecodeExpertHistogram, TokenBoundaryCanBeBeforeNonRoutedTailLayer)
{
    auto cfg = makeConfig(3, 8, 2, 2);
    cfg.token_boundary_layer_idx = 1;
    DecodeExpertHistogram hist(cfg);

    int idx[] = {0, 1};
    float weights[] = {0.5f, 0.5f};

    hist.record(0, idx, weights, 2);
    EXPECT_EQ(hist.windowTokenCount(), 0u);

    hist.record(1, idx, weights, 2);
    EXPECT_EQ(hist.windowTokenCount(), 1u);
    EXPECT_FALSE(hist.windowFull());

    hist.record(0, idx, weights, 2);
    hist.record(1, idx, weights, 2);
    EXPECT_EQ(hist.windowTokenCount(), 2u);
    EXPECT_TRUE(hist.windowFull());

    hist.recordTokenBoundary(2);
    EXPECT_EQ(hist.windowTokenCount(), 2u)
        << "Non-routed tail layers must not advance the routed decode window.";
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

TEST(Test__DecodeExpertHistogram, UpdateOwnershipChangesExactLayerMapping)
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
    const auto new_ownership = makeOwnership(1, 2, {1, 1, 0, 0});
    hist.updateOwnership(new_ownership);

    // After update: same counts, but expert 0 is now on socket 1
    auto loads_after = hist.socketLoads(0);
    EXPECT_EQ(loads_after[0], 0u);
    EXPECT_EQ(loads_after[1], 100u);
    EXPECT_EQ(hist.config().ownership, new_ownership);
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

TEST(Test__DecodeExpertHistogram, PlacementImbalanceScoresArbitraryPlacement)
{
    auto cfg = makeConfig(1, 4, 1, 256, 2);
    DecodeExpertHistogram hist(cfg);

    const uint64_t counts[] = {10, 8, 1, 1};
    hist.mergeLayerCounts(0, counts, 4);

    const auto overloaded = hist.placementImbalance(
        makeOwnership(1, 2, {0, 0, 1, 1}));
    const auto balanced = hist.placementImbalance(
        makeOwnership(1, 2, {0, 1, 1, 0}));

    ASSERT_TRUE(overloaded.valid);
    ASSERT_TRUE(balanced.valid);
    EXPECT_EQ(overloaded.layer_count, 1);
    EXPECT_EQ(overloaded.active_layer_count, 1);
    EXPECT_EQ(overloaded.total_activations, 20u);
    EXPECT_EQ(overloaded.worst_layer, 0);
    EXPECT_NEAR(overloaded.average_spread, 16.0 / 18.0, 1e-9);
    EXPECT_NEAR(balanced.average_spread, 2.0 / 11.0, 1e-9);
    EXPECT_GT(overloaded.average_spread, balanced.average_spread);
}

TEST(Test__DecodeExpertHistogram, CurrentPlacementImbalanceFollowsOwnershipUpdates)
{
    auto cfg = makeConfig(1, 4, 1, 256, 2);
    DecodeExpertHistogram hist(cfg);

    const uint64_t counts[] = {10, 8, 1, 1};
    hist.mergeLayerCounts(0, counts, 4);

    const auto initial = hist.currentPlacementImbalance();
    hist.updateOwnership(makeOwnership(1, 2, {0, 0, 1, 1}));
    const auto overloaded = hist.currentPlacementImbalance();

    ASSERT_TRUE(initial.valid);
    ASSERT_TRUE(overloaded.valid);
    EXPECT_LT(initial.average_spread, overloaded.average_spread);
    EXPECT_EQ(overloaded.worst_layer, 0);
}

TEST(Test__DecodeExpertHistogram, OwnershipUpdateIsLayerExact)
{
    auto cfg = makeConfig(2, 4, 1, 256, 2);
    DecodeExpertHistogram hist(cfg);

    const uint64_t counts[] = {10, 1, 1, 1};
    hist.mergeLayerCounts(0, counts, 4);
    hist.mergeLayerCounts(1, counts, 4);

    // Move expert zero only in layer one. Layer zero must retain the original
    // round-robin ownership; a global expert vector could not represent this.
    const MoELayeredExpertOwnership layered(
        2,
        {
            {0, 1, 0, 1},
            {1, 1, 0, 0},
        });
    hist.updateOwnership(layered);

    EXPECT_EQ(hist.socketLoads(0), (std::vector<uint64_t>{11, 2}));
    EXPECT_EQ(hist.socketLoads(1), (std::vector<uint64_t>{2, 11}));
    EXPECT_EQ(hist.config().ownership, layered);
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

TEST(Test__DecodeExpertHistogram, FrozenGenerationSurvivesNewInferenceWrites)
{
    auto cfg = makeConfig(1, 4, 1, 5);
    DecodeExpertHistogram hist(cfg);
    int old_expert[] = {0};
    int new_expert[] = {1};
    float weight[] = {1.0f};

    for (int token = 0; token < 5; ++token)
        hist.record(0, old_expert, weight, 1);

    const auto frozen = hist.freezeAndRotateWindow();
    ASSERT_TRUE(frozen.valid());
    EXPECT_EQ(frozen.generation, 0u);
    EXPECT_EQ(frozen.token_count, 5u);
    EXPECT_EQ(frozen.activationCount(0, 0), 5u);
    EXPECT_EQ(hist.windowGeneration(), 1u);
    EXPECT_EQ(hist.windowTokenCount(), 0u);

    for (int token = 0; token < 3; ++token)
        hist.record(0, new_expert, weight, 1);

    EXPECT_EQ(hist.activationCount(0, 0), 0u);
    EXPECT_EQ(hist.activationCount(0, 1), 3u);
    EXPECT_EQ(hist.windowTokenCount(), 3u);
    EXPECT_EQ(frozen.activationCount(0, 0), 5u)
        << "A long migration retains immutable evidence from its generation";
    EXPECT_EQ(frozen.activationCount(0, 1), 0u);
}

TEST(Test__DecodeExpertHistogram, ConcurrentRotationLosesNoInferenceRecords)
{
    auto cfg = makeConfig(1, 4, 1, 64);
    DecodeExpertHistogram hist(cfg);
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> completed_records{0};
    int expert[] = {2};
    float weight[] = {1.0f};

    std::thread inference_writer([&]
                                 {
        while (!stop.load(std::memory_order_acquire))
        {
            hist.record(0, expert, weight, 1);
            completed_records.fetch_add(1, std::memory_order_release);
        } });

    while (completed_records.load(std::memory_order_acquire) < 1000)
        std::this_thread::yield();

    uint64_t frozen_records = 0;
    for (int generation = 0; generation < 32; ++generation)
    {
        const auto window = hist.freezeAndRotateWindow();
        ASSERT_TRUE(window.valid());
        frozen_records += window.activationCount(0, 2);
    }

    stop.store(true, std::memory_order_release);
    inference_writer.join();
    const uint64_t expected =
        completed_records.load(std::memory_order_acquire);
    const auto final_window = hist.freezeAndRotateWindow();
    frozen_records += final_window.activationCount(0, 2);

    EXPECT_EQ(frozen_records, expected)
        << "Each record belongs to exactly one RCU histogram generation";
    EXPECT_GT(expected, 1000u)
        << "Inference must continue making progress during maintenance rotation";
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

TEST(Test__DecodeExpertHistogram,
     AsyncDrainGenerationDoesNotRestartReadySourcesWhilePeerIsPending)
{
    auto cfg = makeConfig(1, 4, 1, 1);
    DecodeExpertHistogram hist(cfg);

    int immediate_calls = 0;
    int delayed_calls = 0;
    hist.registerRuntimeHistogramDrain(
        [&]()
        {
            ++immediate_calls;
            const uint64_t counts[4] = {1, 0, 0, 0};
            hist.mergeLayerCounts(0, counts, 4);
            return RuntimeExpertHistogramDrainResult::ready();
        });
    hist.registerRuntimeHistogramDrain(
        [&]()
        {
            ++delayed_calls;
            if (delayed_calls == 1)
                return RuntimeExpertHistogramDrainResult::pending();
            const uint64_t counts[4] = {0, 0, 3, 0};
            hist.mergeLayerCounts(0, counts, 4);
            return RuntimeExpertHistogramDrainResult::ready();
        });

    const auto first = hist.progressRuntimeHistogramDrains();
    EXPECT_EQ(
        first.progress,
        RuntimeExpertHistogramDrainProgress::Pending);
    EXPECT_EQ(immediate_calls, 1);
    EXPECT_EQ(delayed_calls, 1);
    EXPECT_THROW(
        (void)hist.syncRuntimeHistograms(),
        std::logic_error)
        << "A blocking sync must not cross an active async generation";

    const auto second = hist.progressRuntimeHistogramDrains();
    EXPECT_EQ(second.progress, RuntimeExpertHistogramDrainProgress::Ready);
    EXPECT_EQ(immediate_calls, 1)
        << "A ready source must not begin generation N+1 early";
    EXPECT_EQ(delayed_calls, 2);
    EXPECT_EQ(hist.activationCount(0, 0), 1u);
    EXPECT_EQ(hist.activationCount(0, 2), 3u);

    const auto third = hist.progressRuntimeHistogramDrains();
    EXPECT_EQ(third.progress, RuntimeExpertHistogramDrainProgress::Ready);
    EXPECT_EQ(immediate_calls, 2)
        << "All sources may begin again only after the prior aggregate completed";
    EXPECT_EQ(delayed_calls, 3);
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
    std::vector<std::uint64_t> count_scratch(
        static_cast<std::size_t>(cfg.num_experts));

    auto result = hist.mergeRoutedExpertRows(
        chunk0_routes.data(), chunk0, count_scratch);

    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.tokens_counted, 3u);
    EXPECT_EQ(result.activations_merged, 6u);
    EXPECT_EQ(hist.windowTokenCount(), 3u);
    EXPECT_EQ(hist.activationCount(1, 0), 1u);
    EXPECT_EQ(hist.activationCount(1, 1), 2u);
    EXPECT_EQ(hist.activationCount(1, 2), 2u);
    EXPECT_EQ(hist.activationCount(1, 3), 1u);
    EXPECT_EQ(hist.activationCount(1, 7), 0u);
    EXPECT_EQ(
        hist.activationCount(ExpertHistogramSource::PrefillChunk, 1, 1),
        2u);
    EXPECT_EQ(
        hist.activationCount(ExpertHistogramSource::DecodeToken, 1, 1),
        0u);

    const std::vector<int> chunk1_routes = {
        4, 5,
        5, 6,
        99, 99,
        99, 99};
    RoutedExpertHistogramMerge chunk1 = chunk0;
    chunk1.real_token_count = 2;

    result = hist.mergeRoutedExpertRows(
        chunk1_routes.data(), chunk1, count_scratch);

    ASSERT_TRUE(result) << result.error;
    EXPECT_EQ(result.tokens_counted, 2u);
    EXPECT_EQ(result.activations_merged, 4u);
    EXPECT_EQ(hist.windowTokenCount(), 5u);
    EXPECT_EQ(hist.activationCount(1, 4), 1u);
    EXPECT_EQ(hist.activationCount(1, 5), 2u);
    EXPECT_EQ(hist.activationCount(1, 6), 1u);
    EXPECT_EQ(hist.activationCount(1, 7), 0u);
}

TEST(Test__DecodeExpertHistogram, FrozenWindowRetainsExactProductionPhaseEvidence)
{
    auto cfg = makeConfig(1, 4, 2, 16);
    DecodeExpertHistogram hist(cfg);

    const int decode_experts[2] = {0, 1};
    const float decode_weights[2] = {0.75f, 0.25f};
    hist.record(0, decode_experts, decode_weights, 2);

    const std::vector<int> prefill_routes = {
        1, 2,
        1, 3,
    };
    std::vector<std::uint64_t> count_scratch(
        static_cast<std::size_t>(cfg.num_experts));
    const auto prefill_result = hist.mergeRoutedExpertRows(
        prefill_routes.data(),
        RoutedExpertHistogramMerge{
            .source = ExpertHistogramSource::PrefillChunk,
            .layer_idx = 0,
            .real_token_count = 2,
            .bucket_token_count = 2,
            .top_k = 2,
            .route_stride = 2,
            .count_window_tokens = true,
        },
        count_scratch);
    ASSERT_TRUE(prefill_result) << prefill_result.error;

    const std::uint64_t verifier_counts[4] = {0, 0, 3, 1};
    hist.mergeLayerCounts(
        0,
        verifier_counts,
        4,
        /*count_window_tokens=*/true,
        ExpertHistogramSource::GroupedVerifier);

    EXPECT_EQ(hist.windowTokenCount(), 5u);
    EXPECT_EQ(hist.activationCount(0, 0), 1u);
    EXPECT_EQ(hist.activationCount(0, 1), 3u);
    EXPECT_EQ(hist.activationCount(0, 2), 4u);
    EXPECT_EQ(hist.activationCount(0, 3), 2u);
    EXPECT_EQ(
        hist.layerHistogram(ExpertHistogramSource::DecodeToken, 0),
        (std::vector<std::uint64_t>{1, 1, 0, 0}));
    EXPECT_EQ(
        hist.layerHistogram(ExpertHistogramSource::PrefillChunk, 0),
        (std::vector<std::uint64_t>{0, 2, 1, 1}));
    EXPECT_EQ(
        hist.layerHistogram(ExpertHistogramSource::GroupedVerifier, 0),
        (std::vector<std::uint64_t>{0, 0, 3, 1}));

    const auto frozen = hist.freezeAndRotateWindow();
    ASSERT_TRUE(frozen.valid());
    EXPECT_EQ(
        frozen.source_token_counts,
        (std::array<std::uint64_t, kExpertHistogramProductionSourceCount>{
            1, 2, 2}));
    EXPECT_EQ(
        frozen.layerHistogram(ExpertHistogramSource::PrefillChunk, 0),
        (std::vector<std::uint64_t>{0, 2, 1, 1}));
    EXPECT_THROW(
        (void)frozen.activationCount(
            ExpertHistogramSource::SyntheticTest, 0, 0),
        std::invalid_argument);
}

TEST(Test__DecodeExpertHistogram,
     ValidatedFrozenViewAuthenticatesOnceAndProvidesTypedConstantTimeReads)
{
    auto cfg = makeConfig(2, 4, 2, 8);
    DecodeExpertHistogram hist(cfg);
    const std::array<int, 2> decode_routes{1, 3};
    std::vector<std::uint64_t> count_scratch(
        static_cast<std::size_t>(cfg.num_experts));
    const auto merge = hist.mergeRoutedExpertRows(
        decode_routes.data(),
        RoutedExpertHistogramMerge{
            .source = ExpertHistogramSource::DecodeToken,
            .layer_idx = 1,
            .real_token_count = 1,
            .bucket_token_count = 1,
            .top_k = 2,
            .route_stride = 2,
            .count_window_tokens = true,
        },
        count_scratch);
    ASSERT_TRUE(merge) << merge.error;

    const auto frozen = hist.freezeAndRotateWindow();
    const auto view = frozen.validatedView();
    EXPECT_EQ(view.numLayers(), 2);
    EXPECT_EQ(view.numExperts(), 4);
    EXPECT_EQ(view.generation(), frozen.generation);
    EXPECT_EQ(view.tokenCount(), frozen.token_count);
    EXPECT_THROW((void)view.transactionDemand(), std::logic_error)
        << "Counts-only windows may rank placement, but cannot price concurrent service";
    EXPECT_EQ(view.activationCount(1, 1), 1u);
    EXPECT_EQ(
        view.activationCount(
            ExpertHistogramSource::DecodeToken, 1, 3),
        1u);
    EXPECT_THROW((void)view.activationCount(2, 0), std::out_of_range);
    EXPECT_THROW(
        (void)view.activationCount(
            ExpertHistogramSource::SyntheticTest, 1, 1),
        std::invalid_argument);

    auto malformed = frozen;
    ++malformed.source_expert_counts.front();
    EXPECT_FALSE(malformed.valid());
    EXPECT_THROW((void)malformed.validatedView(), std::invalid_argument);
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
    std::vector<std::uint64_t> count_scratch(
        static_cast<std::size_t>(cfg.num_experts));

    auto result = hist.mergeRoutedExpertRows(
        routes.data(), merge, count_scratch);

    EXPECT_FALSE(result);
    EXPECT_NE(result.error.find("expert id"), std::string::npos);
    EXPECT_EQ(hist.windowTokenCount(), 0u);
    for (int expert = 0; expert < cfg.num_experts; ++expert)
        EXPECT_EQ(hist.activationCount(0, expert), 0u);
}

TEST(Test__DecodeExpertHistogram,
     RoutedRowMergeRejectsUndersizedCallerScratchBeforeMutation)
{
    auto cfg = makeConfig(1, 4, 2, 8);
    DecodeExpertHistogram hist(cfg);
    const std::array<int, 2> routes{0, 1};
    std::array<std::uint64_t, 3> undersized_scratch{};

    const auto result = hist.mergeRoutedExpertRows(
        routes.data(),
        RoutedExpertHistogramMerge{
            .source = ExpertHistogramSource::DecodeToken,
            .layer_idx = 0,
            .real_token_count = 1,
            .bucket_token_count = 1,
            .top_k = 2,
            .route_stride = 2,
            .count_window_tokens = true,
        },
        undersized_scratch);

    EXPECT_FALSE(result);
    EXPECT_NE(result.error.find("expert_count_scratch"), std::string::npos);
    EXPECT_EQ(hist.windowTokenCount(), 0u);
    for (int expert = 0; expert < cfg.num_experts; ++expert)
        EXPECT_EQ(hist.activationCount(0, expert), 0u);
}
