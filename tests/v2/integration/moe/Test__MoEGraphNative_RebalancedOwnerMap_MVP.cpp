/**
 * @file Test__MoEGraphNative_RebalancedOwnerMap_MVP.cpp
 * @brief Phase 13 integration test — RoutedTierRebalanced owner map validation.
 *
 * Model-light config-only test: plans a rebalanced owner map and validates:
 *  - Exactly one canonical owner per expert per layer.
 *  - Tier diagnostics are populated and consistent.
 *  - All-GPU plan produces no CPU owners.
 *  - Mixed GPU/CPU plan: GPU-tier experts match histogram ordering.
 *
 * No model loading, no weight I/O, single MPI rank.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEExpertParallelPlanner.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kLayers = 4;
        constexpr int kExperts = 16;
        constexpr int kDModel = 32;
        constexpr int kIntermediate = 16;

        // -------------------------------------------------------------------------
        // Domain helpers (same style as unit tests)
        // -------------------------------------------------------------------------

        ExpertComputeDomain cudaDomain(const std::string &name)
        {
            ExpertComputeDomain d;
            d.name = name;
            d.kind = ExpertDomainKind::SingleDevice;
            d.backend = CollectiveBackendType::NCCL;
            d.participants = {GlobalDeviceAddress::cuda(0, 0)};
            d.owner_rank = 0;
            d.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return d;
        }

        ExpertComputeDomain rocmDomain(const std::string &name)
        {
            ExpertComputeDomain d;
            d.name = name;
            d.kind = ExpertDomainKind::LocalTP;
            d.backend = CollectiveBackendType::RCCL;
            d.participants = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(0, 1)};
            d.owner_rank = 0;
            d.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return d;
        }

        ExpertComputeDomain cpuDomain(const std::string &name)
        {
            ExpertComputeDomain d;
            d.name = name;
            d.kind = ExpertDomainKind::SingleDevice;
            d.backend = CollectiveBackendType::MPI;
            d.participants = {GlobalDeviceAddress::cpu(0)};
            d.owner_rank = 0;
            d.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return d;
        }

        ExpertRoutedTier tier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int max_experts_per_layer,
            bool fallback = false)
        {
            ExpertRoutedTier t;
            t.name = name;
            t.domain = domain;
            t.priority = priority;
            t.max_experts_per_layer = max_experts_per_layer;
            t.fallback = fallback;
            return t;
        }

        MoEExpertModelMetadata metadata()
        {
            MoEExpertModelMetadata m;
            m.num_layers = kLayers;
            m.num_experts = kExperts;
            m.d_model = kDModel;
            m.routed_intermediate_size = kIntermediate;
            m.has_shared_expert = false;
            m.routed_quant_type = "F32";
            return m;
        }

        // Build a histogram: each layer has ascending activation counts 1..kExperts.
        // Returns unique_ptr because DecodeExpertHistogram is not movable (atomics).
        std::unique_ptr<DecodeExpertHistogram> makeAscendingHistogram()
        {
            DecodeExpertHistogramConfig cfg;
            cfg.num_layers = kLayers;
            cfg.num_experts = kExperts;
            cfg.top_k = 2;
            cfg.window_size = 512;
            auto hist = std::make_unique<DecodeExpertHistogram>(cfg);
            for (int layer = 0; layer < kLayers; ++layer)
            {
                for (int expert = 0; expert < kExperts; ++expert)
                {
                    const uint64_t count = static_cast<uint64_t>(expert + 1);
                    const float weight = 1.0f;
                    for (uint64_t i = 0; i < count; ++i)
                        hist->record(layer, &expert, &weight, 1);
                }
            }
            return hist;
        }

        void assertExactlyOneOwnerPerExpert(const MoEExpertOwnerMap &owner_map)
        {
            for (int layer = 0; layer < kLayers; ++layer)
                for (int expert = 0; expert < kExperts; ++expert)
                {
                    ASSERT_EQ(owner_map.ownerCountForExpert(layer, expert), 1u)
                        << "layer=" << layer << " expert=" << expert;
                    ASSERT_NE(owner_map.ownerFor(layer, expert), nullptr)
                        << "layer=" << layer << " expert=" << expert;
                }
        }

    } // namespace

    // =============================================================================
    // Integration: all-GPU rebalanced owner map with histogram
    // =============================================================================

    TEST(Test__MoEGraphNative_RebalancedOwnerMap_MVP, AllGPU_CompleteOwnerMap_NoFallback)
    {
        // cuda_hot (8 slots, priority 0) + rocm_warm (8 slots, priority 1) = 16 == kExperts.
        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.continuation_domain = "cuda_hot";
        plan.shared_expert_domain = "cuda_hot";
        plan.residency_policy = ExpertResidencyPolicy::RoutedTierRebalanced;
        plan.domains = {cudaDomain("cuda_hot"), rocmDomain("rocm_warm")};
        plan.routed_tiers = {
            tier("cuda_hot_tier", "cuda_hot", 0, 8),
            tier("rocm_warm_tier", "rocm_warm", 1, 8),
        };

        auto hist = makeAscendingHistogram();
        MoEExpertParallelPlannerOptions opts;
        opts.decode_histogram = hist.get();

        const auto result = MoEExpertParallelPlanner::plan(plan, metadata(), opts);

        // Plan validates cleanly.
        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        result.planned_plan,
                        {.layer_count = kLayers, .routed_expert_count = kExperts})
                        .ok());

        // Exactly kLayers placements.
        ASSERT_EQ(static_cast<int>(result.planned_plan.placements.size()), kLayers);

        // Owner map: exactly one owner per expert, no CPU owners.
        const auto owner_map = MoEExpertOwnerMap::build(result.planned_plan);
        assertExactlyOneOwnerPerExpert(owner_map);

        for (int layer = 0; layer < kLayers; ++layer)
            for (int expert = 0; expert < kExperts; ++expert)
            {
                const auto *owner = owner_map.ownerFor(layer, expert);
                ASSERT_NE(owner, nullptr);
                EXPECT_FALSE(owner->device.is_cpu())
                    << "layer=" << layer << " expert=" << expert
                    << " must not be on CPU in all-GPU plan";
            }

        // Diagnostics: histogram used, GPU coverage 100%, no fallback rows.
        EXPECT_TRUE(result.rebalance_diagnostics.histogram_used);
        ASSERT_EQ(static_cast<int>(result.rebalance_diagnostics.layers.size()), kLayers);
        for (const auto &ld : result.rebalance_diagnostics.layers)
        {
            EXPECT_FLOAT_EQ(ld.gpu_coverage_ratio, 1.0f)
                << "layer=" << ld.layer;
            EXPECT_FLOAT_EQ(ld.expected_cpu_fallback_rows, 0.0f)
                << "layer=" << ld.layer;
            ASSERT_EQ(static_cast<int>(ld.tier_expert_counts.size()), 2);
            // Each tier should own exactly 8 experts.
            EXPECT_EQ(ld.tier_expert_counts[0], 8) << "cuda_hot_tier layer=" << ld.layer;
            EXPECT_EQ(ld.tier_expert_counts[1], 8) << "rocm_warm_tier layer=" << ld.layer;
            EXPECT_GT(ld.gpu_tier_memory_bytes, 0u) << "layer=" << ld.layer;
        }
        EXPECT_FLOAT_EQ(result.rebalance_diagnostics.avg_gpu_coverage_ratio, 1.0f);
        EXPECT_FLOAT_EQ(result.rebalance_diagnostics.avg_cpu_fallback_rows, 0.0f);

        // Hottest expert (kExperts-1 == 15) should land in cuda_hot_tier (priority 0).
        for (int layer = 0; layer < kLayers; ++layer)
        {
            const auto *owner = owner_map.ownerFor(layer, kExperts - 1);
            ASSERT_NE(owner, nullptr);
            EXPECT_EQ(owner->tier_name, "cuda_hot_tier")
                << "Hottest expert should be in highest-priority tier, layer=" << layer;
        }
    }

    // =============================================================================
    // Integration: mixed GPU/CPU rebalanced owner map
    // =============================================================================

    TEST(Test__MoEGraphNative_RebalancedOwnerMap_MVP, MixedGpuCpu_TierDiagnostics_Complete)
    {
        // cuda_hot (4) + rocm_warm (4) = 8 GPU; remaining 8 -> cpu_cold (fallback).
        MoEExpertParallelPlan plan;
        plan.enabled = true;
        plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
        plan.continuation_domain = "cuda_hot";
        plan.shared_expert_domain = "cuda_hot";
        plan.residency_policy = ExpertResidencyPolicy::RoutedTierRebalanced;
        plan.domains = {cudaDomain("cuda_hot"), rocmDomain("rocm_warm"), cpuDomain("cpu_cold")};
        plan.routed_tiers = {
            tier("cuda_hot_tier", "cuda_hot", 0, 4),
            tier("rocm_warm_tier", "rocm_warm", 1, 4),
            tier("cpu_cold_tier", "cpu_cold", 2, 0, true),
        };

        auto hist = makeAscendingHistogram();
        MoEExpertParallelPlannerOptions opts;
        opts.decode_histogram = hist.get();

        const auto result = MoEExpertParallelPlanner::plan(plan, metadata(), opts);

        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        result.planned_plan,
                        {.layer_count = kLayers, .routed_expert_count = kExperts})
                        .ok());

        const auto owner_map = MoEExpertOwnerMap::build(result.planned_plan);
        assertExactlyOneOwnerPerExpert(owner_map);

        // Diagnostics: 3 tiers, GPU coverage == 0.5.
        ASSERT_EQ(static_cast<int>(result.rebalance_diagnostics.layers.size()), kLayers);
        for (const auto &ld : result.rebalance_diagnostics.layers)
        {
            EXPECT_FLOAT_EQ(ld.gpu_coverage_ratio, 0.5f) << "layer=" << ld.layer;
            EXPECT_GT(ld.expected_cpu_fallback_rows, 0.0f) << "layer=" << ld.layer;
            ASSERT_EQ(static_cast<int>(ld.tier_expert_counts.size()), 3);
            EXPECT_EQ(ld.tier_expert_counts[0], 4) << "cuda_hot_tier count, layer=" << ld.layer;
            EXPECT_EQ(ld.tier_expert_counts[1], 4) << "rocm_warm_tier count, layer=" << ld.layer;
            EXPECT_EQ(ld.tier_expert_counts[2], 8) << "cpu_cold_tier fallback count, layer=" << ld.layer;
        }
        EXPECT_FLOAT_EQ(result.rebalance_diagnostics.avg_gpu_coverage_ratio, 0.5f);

        // Hottest experts (15..8) should be on GPU; coldest (7..0) should be on CPU fallback.
        for (int layer = 0; layer < kLayers; ++layer)
        {
            for (int hot = kExperts / 2; hot < kExperts; ++hot)
            {
                const auto *owner = owner_map.ownerFor(layer, hot);
                ASSERT_NE(owner, nullptr);
                EXPECT_NE(owner->domain_name, "cpu_cold")
                    << "Hot expert " << hot << " should not be on CPU, layer=" << layer;
            }
            for (int cold = 0; cold < kExperts / 2; ++cold)
            {
                const auto *owner = owner_map.ownerFor(layer, cold);
                ASSERT_NE(owner, nullptr);
                EXPECT_EQ(owner->domain_name, "cpu_cold")
                    << "Cold expert " << cold << " should be on CPU fallback, layer=" << layer;
            }
        }
    }

} // namespace llaminar2::test

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
