/**
 * @file Test__MoERoutedTierRebalancer.cpp
 * @brief Phase 13 unit tests for MoEExpertParallelPlanner RoutedTierRebalanced policy.
 *
 * Tests:
 *  - AllGPU: all-GPU cuda_hot/rocm_warm plan with histogram; no CPU fallback.
 *  - MixedGpuCpu: cuda_hot/rocm_warm/cpu_cold plan; hottest go to GPU,
 *    cold go to CPU fallback; increasing GPU capacity reduces fallback assignment.
 */

#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEExpertParallelPlanner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kLayers = 2;
        constexpr int kExperts = 8;
        constexpr int kDModel = 16;
        constexpr int kIntermediate = 8;

        // -------------------------------------------------------------------------
        // Domain helpers
        // -------------------------------------------------------------------------

        ExpertComputeDomain cudaDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain rocmDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(0, 1)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain cpuDomain(const std::string &name)
        {
            ExpertComputeDomain domain;
            domain.name = name;
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier tier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int max_experts_per_layer,
            bool fallback = false)
        {
            ExpertRoutedTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.max_experts_per_layer = max_experts_per_layer;
            result.fallback = fallback;
            return result;
        }

        MoEExpertModelMetadata metadata()
        {
            MoEExpertModelMetadata model;
            model.num_layers = kLayers;
            model.num_experts = kExperts;
            model.d_model = kDModel;
            model.routed_intermediate_size = kIntermediate;
            model.has_shared_expert = false;
            model.routed_quant_type = "F32";
            return model;
        }

        // All-GPU plan: cuda_hot + rocm_warm, no fallback, total capacity == kExperts.
        MoEExpertParallelPlan allGpuPlan(int cuda_capacity = 4, int rocm_capacity = 4)
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy = ExpertResidencyPolicy::RoutedTierRebalanced;
            plan.domains = {cudaDomain("cuda_hot"), rocmDomain("rocm_warm")};
            plan.routed_tiers = {
                tier("cuda_hot_tier", "cuda_hot", 0, cuda_capacity),   // highest priority (0)
                tier("rocm_warm_tier", "rocm_warm", 1, rocm_capacity), // lower priority (1)
            };
            return plan;
        }

        // Mixed GPU+CPU plan: cuda_hot + rocm_warm + cpu_cold (fallback).
        MoEExpertParallelPlan mixedPlan(int cuda_capacity, int rocm_capacity)
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy = ExpertResidencyPolicy::RoutedTierRebalanced;
            plan.domains = {cudaDomain("cuda_hot"), rocmDomain("rocm_warm"), cpuDomain("cpu_cold")};
            plan.routed_tiers = {
                tier("cuda_hot_tier", "cuda_hot", 0, cuda_capacity),
                tier("rocm_warm_tier", "rocm_warm", 1, rocm_capacity),
                tier("cpu_cold_tier", "cpu_cold", 2, 0, true), // fallback
            };
            return plan;
        }

        // Build a DecodeExpertHistogram with custom counts.
        // counts[layer][expert] drives activation ordering.
        // Returns unique_ptr because DecodeExpertHistogram is not movable (atomics).
        std::unique_ptr<DecodeExpertHistogram> makeHistogram(const std::vector<std::vector<uint64_t>> &counts)
        {
            DecodeExpertHistogramConfig cfg;
            cfg.num_layers = static_cast<int>(counts.size());
            cfg.num_experts = counts.empty() ? 0 : static_cast<int>(counts[0].size());
            cfg.top_k = 2;
            cfg.window_size = 256;
            auto hist = std::make_unique<DecodeExpertHistogram>(cfg);

            for (int layer = 0; layer < cfg.num_layers; ++layer)
            {
                for (int expert = 0; expert < cfg.num_experts; ++expert)
                {
                    const uint64_t count = counts[static_cast<size_t>(layer)][static_cast<size_t>(expert)];
                    const int eidx = expert;
                    const float weight = 1.0f;
                    for (uint64_t i = 0; i < count; ++i)
                        hist->record(layer, &eidx, &weight, 1);
                }
            }
            return hist;
        }

        void expectExactlyOneOwnerPerExpert(const MoEExpertOwnerMap &owner_map, int num_layers, int num_experts)
        {
            for (int layer = 0; layer < num_layers; ++layer)
            {
                for (int expert = 0; expert < num_experts; ++expert)
                {
                    EXPECT_EQ(owner_map.ownerCountForExpert(layer, expert), 1u)
                        << "layer=" << layer << " expert=" << expert;
                    EXPECT_NE(owner_map.ownerFor(layer, expert), nullptr)
                        << "layer=" << layer << " expert=" << expert;
                }
            }
        }

        int countFallbackOwners(const MoEExpertOwnerMap &owner_map, int num_layers, int num_experts, const std::string &fallback_domain)
        {
            int count = 0;
            for (int layer = 0; layer < num_layers; ++layer)
                for (int expert = 0; expert < num_experts; ++expert)
                {
                    const auto *owner = owner_map.ownerFor(layer, expert);
                    if (owner && owner->domain_name == fallback_domain)
                        ++count;
                }
            return count;
        }

    } // namespace

    // =============================================================================
    // V2_Unit_MoERoutedTierRebalancer_AllGPU
    // =============================================================================

    TEST(Test__MoERoutedTierRebalancer, AllGPU_FullCapacity_NoFallback)
    {
        // With histogram: experts 7..0 ordered by descending activation count.
        // Layer 0: expert 7 hottest, expert 0 coldest.
        // Layer 1: expert 0 hottest, expert 7 coldest (inverted).
        std::vector<std::vector<uint64_t>> counts(kLayers, std::vector<uint64_t>(kExperts, 0));
        for (int e = 0; e < kExperts; ++e)
        {
            counts[0][static_cast<size_t>(e)] = static_cast<uint64_t>(kExperts - e); // 8,7,6,5,4,3,2,1
            counts[1][static_cast<size_t>(e)] = static_cast<uint64_t>(e + 1);        // 1,2,3,4,5,6,7,8
        }
        auto hist = makeHistogram(counts);

        MoEExpertParallelPlannerOptions opts;
        opts.decode_histogram = hist.get();

        const auto result = MoEExpertParallelPlanner::plan(allGpuPlan(), metadata(), opts);

        // Plan must be valid.
        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        result.planned_plan,
                        {.layer_count = kLayers, .routed_expert_count = kExperts})
                        .ok());

        // Exactly kLayers placements.
        ASSERT_EQ(static_cast<int>(result.planned_plan.placements.size()), kLayers);

        // Owner map: one owner per expert, no CPU owners.
        const auto owner_map = MoEExpertOwnerMap::build(result.planned_plan);
        expectExactlyOneOwnerPerExpert(owner_map, kLayers, kExperts);
        for (int layer = 0; layer < kLayers; ++layer)
            for (int expert = 0; expert < kExperts; ++expert)
            {
                const auto *owner = owner_map.ownerFor(layer, expert);
                ASSERT_NE(owner, nullptr);
                EXPECT_FALSE(owner->device.is_cpu())
                    << "layer=" << layer << " expert=" << expert
                    << " should not be on CPU in all-GPU plan";
            }

        // No fallback assignments: each tier has exactly 4 experts.
        const auto &placement0 = result.planned_plan.placements[0];
        const auto &tiers0 = placement0.routed_expert_tier;
        ASSERT_EQ(static_cast<int>(tiers0.size()), kExperts);
        const int cuda_count = static_cast<int>(std::count(tiers0.begin(), tiers0.end(), 0));
        const int rocm_count = static_cast<int>(std::count(tiers0.begin(), tiers0.end(), 1));
        EXPECT_EQ(cuda_count, 4);
        EXPECT_EQ(rocm_count, 4);

        // Hottest experts (7,6,5,4 in layer 0) should be in cuda_hot_tier (priority 0).
        // Layer 0 histogram: expert 7 hottest => cuda tier.
        {
            const auto *owner_e7 = owner_map.ownerFor(0, 7);
            ASSERT_NE(owner_e7, nullptr);
            EXPECT_EQ(owner_e7->tier_name, "cuda_hot_tier")
                << "Hottest expert should land in highest-priority tier";
        }

        // Diagnostics: histogram was used, GPU coverage ratio == 1.0.
        EXPECT_TRUE(result.rebalance_diagnostics.histogram_used);
        ASSERT_EQ(static_cast<int>(result.rebalance_diagnostics.layers.size()), kLayers);
        for (const auto &ld : result.rebalance_diagnostics.layers)
        {
            EXPECT_FLOAT_EQ(ld.gpu_coverage_ratio, 1.0f);
            EXPECT_FLOAT_EQ(ld.expected_cpu_fallback_rows, 0.0f);
            EXPECT_GT(ld.gpu_tier_memory_bytes, 0u);
            ASSERT_EQ(static_cast<int>(ld.tier_expert_counts.size()), 2); // 2 tiers
        }
        EXPECT_FLOAT_EQ(result.rebalance_diagnostics.avg_gpu_coverage_ratio, 1.0f);
        EXPECT_FLOAT_EQ(result.rebalance_diagnostics.avg_cpu_fallback_rows, 0.0f);

        // No duplicates in tier vector.
        for (const auto &placement : result.planned_plan.placements)
        {
            for (int e = 0; e < kExperts; ++e)
            {
                const int t = placement.routed_expert_tier[static_cast<size_t>(e)];
                EXPECT_GE(t, 0) << "No expert should be unassigned in all-GPU plan";
            }
        }
    }

    TEST(Test__MoERoutedTierRebalancer, AllGPU_InsufficientCapacity_Throws)
    {
        // cuda=2 + rocm=2 = 4 < kExperts=8 with no fallback: must throw.
        const auto plan = allGpuPlan(2, 2);
        EXPECT_THROW(
            (void)MoEExpertParallelPlanner::plan(plan, metadata()),
            std::invalid_argument);

        // With fallback tier it should succeed.
        auto plan_with_fallback = plan;
        plan_with_fallback.domains.push_back(cpuDomain("cpu_cold"));
        plan_with_fallback.routed_tiers.push_back(
            tier("cpu_cold_tier", "cpu_cold", 3, 0, true));
        EXPECT_NO_THROW(
            (void)MoEExpertParallelPlanner::plan(plan_with_fallback, metadata()));
    }

    // =============================================================================
    // V2_Unit_MoERoutedTierRebalancer_MixedGpuCpu
    // =============================================================================

    TEST(Test__MoERoutedTierRebalancer, MixedGpuCpu_HottestGoToGPU_ColdGoToCPU)
    {
        // Layer 0 histogram: experts 7..4 much hotter than experts 3..0.
        std::vector<std::vector<uint64_t>> counts(kLayers, std::vector<uint64_t>(kExperts, 0));
        counts[0] = {1, 2, 3, 4, 100, 200, 300, 400}; // experts 4-7 are hot
        counts[1] = {400, 300, 200, 100, 4, 3, 2, 1}; // experts 0-3 are hot in layer 1

        auto hist = makeHistogram(counts);

        MoEExpertParallelPlannerOptions opts;
        opts.decode_histogram = hist.get();

        // cuda=2 + rocm=2 = 4 GPU slots; remaining 4 go to CPU fallback.
        const auto result = MoEExpertParallelPlanner::plan(mixedPlan(2, 2), metadata(), opts);

        EXPECT_TRUE(validateMoEExpertParallelPlan(
                        result.planned_plan,
                        {.layer_count = kLayers, .routed_expert_count = kExperts})
                        .ok());

        const auto owner_map = MoEExpertOwnerMap::build(result.planned_plan);
        expectExactlyOneOwnerPerExpert(owner_map, kLayers, kExperts);

        // Layer 0: hottest experts (7,6,5,4) should be on GPU.
        for (int hot_expert : {4, 5, 6, 7})
        {
            const auto *owner = owner_map.ownerFor(0, hot_expert);
            ASSERT_NE(owner, nullptr);
            EXPECT_NE(owner->domain_name, "cpu_cold")
                << "Hot expert " << hot_expert << " should not be on CPU in layer 0";
        }
        // Layer 0: coldest experts (0,1,2,3) should be on CPU fallback.
        for (int cold_expert : {0, 1, 2, 3})
        {
            const auto *owner = owner_map.ownerFor(0, cold_expert);
            ASSERT_NE(owner, nullptr);
            EXPECT_EQ(owner->domain_name, "cpu_cold")
                << "Cold expert " << cold_expert << " should be on CPU fallback in layer 0";
        }

        // Layer 1: hottest experts (0,1,2,3) should be on GPU.
        for (int hot_expert : {0, 1, 2, 3})
        {
            const auto *owner = owner_map.ownerFor(1, hot_expert);
            ASSERT_NE(owner, nullptr);
            EXPECT_NE(owner->domain_name, "cpu_cold")
                << "Hot expert " << hot_expert << " should not be on CPU in layer 1";
        }

        // Diagnostics: 4/8 experts on GPU -> gpu_coverage_ratio = 0.5.
        for (const auto &ld : result.rebalance_diagnostics.layers)
        {
            EXPECT_FLOAT_EQ(ld.gpu_coverage_ratio, 0.5f);
            EXPECT_GT(ld.expected_cpu_fallback_rows, 0.0f);
            ASSERT_EQ(static_cast<int>(ld.tier_expert_counts.size()), 3); // 3 tiers
            // CPU fallback tier (index 2) should have 4 experts.
            EXPECT_EQ(ld.tier_expert_counts[2], 4);
        }
    }

    TEST(Test__MoERoutedTierRebalancer, MixedGpuCpu_IncreasingGPUCapacity_ReducesFallback)
    {
        // Uniform histogram (all experts equally hot) — capacity is the only factor.
        std::vector<std::vector<uint64_t>> counts(kLayers, std::vector<uint64_t>(kExperts, 10));
        auto hist = makeHistogram(counts);
        MoEExpertParallelPlannerOptions opts;
        opts.decode_histogram = hist.get();

        auto countCPUFallback = [&](int cuda_cap, int rocm_cap) -> int
        {
            const auto r = MoEExpertParallelPlanner::plan(mixedPlan(cuda_cap, rocm_cap), metadata(), opts);
            const auto owner_map = MoEExpertOwnerMap::build(r.planned_plan);
            expectExactlyOneOwnerPerExpert(owner_map, kLayers, kExperts);
            return countFallbackOwners(owner_map, kLayers, kExperts, "cpu_cold");
        };

        const int fallback_2_2 = countCPUFallback(2, 2); // 4 GPU slots => 4 fallback per layer
        const int fallback_3_3 = countCPUFallback(3, 3); // 6 GPU slots => 2 fallback per layer
        const int fallback_4_4 = countCPUFallback(4, 4); // 8 GPU slots => 0 fallback per layer

        EXPECT_GT(fallback_2_2, fallback_3_3)
            << "Increasing GPU capacity (2+2 -> 3+3) should reduce fallback assignments";
        EXPECT_GT(fallback_3_3, fallback_4_4)
            << "Increasing GPU capacity (3+3 -> 4+4) should reduce fallback assignments";
        EXPECT_EQ(fallback_4_4, 0)
            << "Full GPU capacity should leave zero CPU fallback assignments";
    }

} // namespace llaminar2::test
