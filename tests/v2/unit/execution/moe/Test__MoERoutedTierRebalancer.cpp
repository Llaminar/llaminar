/**
 * @file Test__MoERoutedTierRebalancer.cpp
 * @brief Phase 13 unit tests for MoERoutedExpertPlacementPlanner RoutedTierRebalanced policy.
 *
 * Tests:
 *  - AllGPU: all-GPU cuda_hot/rocm_warm plan with histogram; no CPU fallback.
 *  - MixedGpuCpu: cuda_hot/rocm_warm/cpu_cold plan; hottest go to GPU,
 *    cold go to CPU fallback; increasing GPU capacity reduces fallback assignment.
 */

#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
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

        RoutedExpertDomain cudaDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::NCCL;
            domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain rocmDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0), GlobalDeviceAddress::rocm(0, 1)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertDomain cpuDomain(const std::string &name)
        {
            RoutedExpertDomain domain;
            domain.name = name;
            domain.scope = ExecutionDomainScope::SINGLE;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.owner_rank = 0;
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            return domain;
        }

        RoutedExpertTier tier(
            const std::string &name,
            const std::string &domain,
            int priority,
            int max_experts_per_layer,
            bool fallback = false)
        {
            RoutedExpertTier result;
            result.name = name;
            result.domain = domain;
            result.priority = priority;
            result.max_experts_per_layer = max_experts_per_layer;
            result.fallback = fallback;
            return result;
        }

        MoERoutedExpertModelMetadata metadata()
        {
            MoERoutedExpertModelMetadata model;
            model.num_layers = kLayers;
            model.num_experts = kExperts;
            model.d_model = kDModel;
            model.routed_intermediate_size = kIntermediate;
            model.has_shared_expert = false;
            model.routed_quant_type = "F32";
            return model;
        }

        // All-GPU plan: cuda_hot + rocm_warm, no fallback, total capacity == kExperts.
        MoERoutedExpertPlacementPlan allGpuPlan(int cuda_capacity = 4, int rocm_capacity = 4)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy = RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.domains = {cudaDomain("cuda_hot"), rocmDomain("rocm_warm")};
            plan.routed_tiers = {
                tier("cuda_hot_tier", "cuda_hot", 0, cuda_capacity),   // highest priority (0)
                tier("rocm_warm_tier", "rocm_warm", 1, rocm_capacity), // lower priority (1)
            };
            return plan;
        }

        // Mixed GPU+CPU plan: cuda_hot + rocm_warm + cpu_cold (fallback).
        MoERoutedExpertPlacementPlan mixedPlan(int cuda_capacity, int rocm_capacity)
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy = RoutedExpertResidencyPolicy::RoutedTierRebalanced;
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
            /* Placement ordering needs counts only; bind valid load geometry. */
            cfg.sockets = {DeviceId::cpu()};
            cfg.ownership = MoELayeredExpertOwnership::uniform(
                cfg.num_layers,
                /*participant_count=*/1,
                std::vector<int>(
                    static_cast<size_t>(cfg.num_experts),
                    /*owner=*/0));
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

        /** @brief Build a histogram with independently controlled phase evidence. */
        std::unique_ptr<DecodeExpertHistogram> makePhaseHistogram(
            const std::vector<std::vector<uint64_t>> &decode_counts,
            const std::vector<std::vector<uint64_t>> &prefill_counts,
            const std::vector<std::vector<uint64_t>> &verifier_counts)
        {
            if (decode_counts.size() != prefill_counts.size() ||
                decode_counts.size() != verifier_counts.size() ||
                decode_counts.empty())
            {
                throw std::invalid_argument(
                    "Phase histogram fixture requires matching non-empty layers");
            }
            DecodeExpertHistogramConfig cfg;
            cfg.num_layers = static_cast<int>(decode_counts.size());
            cfg.num_experts = static_cast<int>(decode_counts.front().size());
            cfg.top_k = 2;
            cfg.window_size = 256;
            cfg.sockets = {DeviceId::cpu()};
            cfg.ownership = MoELayeredExpertOwnership::uniform(
                cfg.num_layers,
                /*participant_count=*/1,
                std::vector<int>(
                    static_cast<size_t>(cfg.num_experts),
                    /*owner=*/0));
            auto histogram = std::make_unique<DecodeExpertHistogram>(cfg);
            for (int layer = 0; layer < cfg.num_layers; ++layer)
            {
                const auto merge = [&](
                                       const std::vector<uint64_t> &counts,
                                       ExpertHistogramSource source)
                {
                    if (counts.size() !=
                        static_cast<size_t>(cfg.num_experts))
                    {
                        throw std::invalid_argument(
                            "Phase histogram fixture changed expert geometry");
                    }
                    histogram->mergeLayerCounts(
                        layer,
                        counts.data(),
                        cfg.num_experts,
                        /*count_window_tokens=*/false,
                        source);
                };
                merge(
                    decode_counts[static_cast<size_t>(layer)],
                    ExpertHistogramSource::DecodeToken);
                merge(
                    prefill_counts[static_cast<size_t>(layer)],
                    ExpertHistogramSource::PrefillChunk);
                merge(
                    verifier_counts[static_cast<size_t>(layer)],
                    ExpertHistogramSource::GroupedVerifier);
            }
            return histogram;
        }

        /** @brief Build a total tier/layer profile with caller-selected phase costs. */
        MoERoutedTierServiceProfile serviceProfile(
            const MoERoutedExpertPlacementPlan &plan,
            std::array<uint64_t, kExpertHistogramProductionSourceCount>
                preferred_costs,
            std::array<uint64_t, kExpertHistogramProductionSourceCount>
                less_preferred_costs)
        {
            MoERoutedTierServiceProfile profile;
            profile.identity = "unit-certified-phase-profile-v1";
            for (int tier_index = 0;
                 tier_index < static_cast<int>(plan.routed_tiers.size());
                 ++tier_index)
            {
                const auto &costs = tier_index == 0
                                        ? preferred_costs
                                        : less_preferred_costs;
                for (int layer = 0; layer < kLayers; ++layer)
                {
                    profile.costs.push_back({
                        .tier_index = tier_index,
                        .layer = layer,
                        .nanoseconds_per_activation = costs,
                    });
                }
            }
            return profile;
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
            counts[0][static_cast<size_t>(e)] = static_cast<uint64_t>(e + 1);        // 1,2,3,4,5,6,7,8
            counts[1][static_cast<size_t>(e)] = static_cast<uint64_t>(kExperts - e); // 8,7,6,5,4,3,2,1
        }
        auto hist = makeHistogram(counts);

        MoERoutedExpertPlacementPlannerOptions opts;
        opts.decode_histogram = hist.get();

        const auto result = MoERoutedExpertPlacementPlanner::plan(allGpuPlan(), metadata(), opts);

        // Plan must be valid.
        EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(
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
            (void)MoERoutedExpertPlacementPlanner::plan(plan, metadata()),
            std::invalid_argument);

        // With fallback tier it should succeed.
        auto plan_with_fallback = plan;
        plan_with_fallback.domains.push_back(cpuDomain("cpu_cold"));
        plan_with_fallback.routed_tiers.push_back(
            tier("cpu_cold_tier", "cpu_cold", 3, 0, true));
        EXPECT_NO_THROW(
            (void)MoERoutedExpertPlacementPlanner::plan(plan_with_fallback, metadata()));
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

        MoERoutedExpertPlacementPlannerOptions opts;
        opts.decode_histogram = hist.get();

        // cuda=2 + rocm=2 = 4 GPU slots; remaining 4 go to CPU fallback.
        const auto result = MoERoutedExpertPlacementPlanner::plan(mixedPlan(2, 2), metadata(), opts);

        EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(
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
        MoERoutedExpertPlacementPlannerOptions opts;
        opts.decode_histogram = hist.get();

        auto countCPUFallback = [&](int cuda_cap, int rocm_cap) -> int
        {
            const auto r = MoERoutedExpertPlacementPlanner::plan(mixedPlan(cuda_cap, rocm_cap), metadata(), opts);
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

    TEST(
        Test__MoERoutedTierRebalancer,
        ExactPhaseServiceCostCanOutrankAggregateFrequency)
    {
        auto plan = allGpuPlan();
        std::vector<std::vector<uint64_t>> decode(
            kLayers, std::vector<uint64_t>(kExperts, 0));
        std::vector<std::vector<uint64_t>> prefill = decode;
        std::vector<std::vector<uint64_t>> verifier = decode;
        decode[0] = {100, 90, 80, 70, 60, 50, 40, 0};
        prefill[0][7] = 2;
        auto histogram = makePhaseHistogram(decode, prefill, verifier);

        MoERoutedExpertPlacementPlannerOptions aggregate_options;
        aggregate_options.decode_histogram = histogram.get();
        const auto aggregate = MoERoutedExpertPlacementPlanner::plan(
            plan, metadata(), aggregate_options);
        EXPECT_EQ(
            aggregate.planned_plan.placements[0].routed_expert_tier[7],
            1)
            << "Raw frequency should leave expert 7 outside the preferred quota";

        auto profile = serviceProfile(
            plan,
            /* preferred decode/prefill/verifier ns */ {9, 1, 9},
            /* less-preferred decode/prefill/verifier ns */ {10, 100, 10});
        MoERoutedExpertPlacementPlannerOptions phase_options;
        phase_options.decode_histogram = histogram.get();
        phase_options.phase_service_profile = &profile;
        const auto phase = MoERoutedExpertPlacementPlanner::plan(
            plan, metadata(), phase_options);

        const auto &placement =
            phase.planned_plan.placements[0].routed_expert_tier;
        EXPECT_EQ(placement[7], 0)
            << "Two prefill activations save more measured time than expert 3's seventy decode activations";
        EXPECT_EQ(placement[3], 1);
        EXPECT_TRUE(
            phase.rebalance_diagnostics.phase_service_profile_used);
        EXPECT_EQ(
            phase.rebalance_diagnostics.phase_service_profile_identity,
            profile.identity);
    }

    TEST(
        Test__MoERoutedTierRebalancer,
        ExactServiceTieRetainsAdversarialIncumbentPlacement)
    {
        auto plan = allGpuPlan();
        const auto baseline = MoERoutedExpertPlacementPlanner::plan(
            plan, metadata());
        auto incumbent = baseline.planned_plan.placements;
        for (auto &layer : incumbent)
        {
            std::swap(
                layer.routed_expert_tier.front(),
                layer.routed_expert_tier.back());
        }

        std::vector<std::vector<uint64_t>> uniform(
            kLayers, std::vector<uint64_t>(kExperts, 10));
        std::vector<std::vector<uint64_t>> zero(
            kLayers, std::vector<uint64_t>(kExperts, 0));
        auto histogram = makePhaseHistogram(uniform, zero, zero);
        auto profile = serviceProfile(
            plan,
            /* preferred */ {7, 11, 13},
            /* less preferred */ {7, 11, 13});

        MoERoutedExpertPlacementPlannerOptions options;
        options.decode_histogram = histogram.get();
        options.phase_service_profile = &profile;
        options.rebalancer.previous_placements = incumbent;
        const auto result = MoERoutedExpertPlacementPlanner::plan(
            plan, metadata(), options);
        ASSERT_EQ(
            result.planned_plan.placements.size(),
            incumbent.size());
        for (std::size_t layer = 0; layer < incumbent.size(); ++layer)
        {
            EXPECT_EQ(
                result.planned_plan.placements[layer].layer,
                incumbent[layer].layer);
            EXPECT_EQ(
                result.planned_plan.placements[layer]
                    .routed_expert_tier,
                incumbent[layer].routed_expert_tier)
                << "Equal measured service must prefer the installed placement over needless movement";
        }
    }

    TEST(
        Test__MoERoutedTierRebalancer,
        ExactAssignmentMatchesExhaustiveLexicographicOracle)
    {
        auto plan = allGpuPlan();
        const std::vector<uint64_t> decode_row{13, 2, 17, 5, 11, 3, 19, 7};
        const std::vector<uint64_t> prefill_row{1, 9, 0, 4, 2, 8, 3, 6};
        const std::vector<uint64_t> verifier_row{7, 0, 5, 1, 8, 2, 4, 3};
        std::vector<std::vector<uint64_t>> decode(kLayers, decode_row);
        std::vector<std::vector<uint64_t>> prefill(kLayers, prefill_row);
        std::vector<std::vector<uint64_t>> verifier(
            kLayers, verifier_row);
        auto histogram = makePhaseHistogram(decode, prefill, verifier);
        auto profile = serviceProfile(
            plan,
            /* preferred */ {3, 5, 7},
            /* less preferred */ {13, 19, 23});

        auto incumbent = MoERoutedExpertPlacementPlanner::plan(
                             plan, metadata())
                             .planned_plan.placements;
        std::swap(
            incumbent[0].routed_expert_tier[0],
            incumbent[0].routed_expert_tier[7]);

        MoERoutedExpertPlacementPlannerOptions options;
        options.decode_histogram = histogram.get();
        options.phase_service_profile = &profile;
        options.rebalancer.previous_placements = incumbent;
        const auto result = MoERoutedExpertPlacementPlanner::plan(
            plan, metadata(), options);

        using OracleCost = std::tuple<uint64_t, int, uint64_t>;
        std::optional<OracleCost> best_cost;
        uint32_t best_preferred_mask = 0;
        for (uint32_t preferred_mask = 0;
             preferred_mask < (1u << kExperts);
             ++preferred_mask)
        {
            int preferred_count = 0;
            for (int expert = 0; expert < kExperts; ++expert)
                preferred_count += (preferred_mask >> expert) & 1u;
            if (preferred_count != 4)
                continue;

            uint64_t service = 0;
            int moved = 0;
            uint64_t tie = 0;
            for (int expert = 0; expert < kExperts; ++expert)
            {
                const int tier =
                    ((preferred_mask >> expert) & 1u) != 0 ? 0 : 1;
                const auto &costs = tier == 0
                                        ? profile.costs[0]
                                              .nanoseconds_per_activation
                                        : profile.costs[kLayers]
                                              .nanoseconds_per_activation;
                service += decode_row[static_cast<size_t>(expert)] * costs[0] +
                           prefill_row[static_cast<size_t>(expert)] * costs[1] +
                           verifier_row[static_cast<size_t>(expert)] * costs[2];
                moved += incumbent[0].routed_expert_tier[
                             static_cast<size_t>(expert)] != tier
                             ? 1
                             : 0;
                tie += tier == 0 ? static_cast<uint64_t>(expert) : 0u;
            }
            const OracleCost candidate{service, moved, tie};
            if (!best_cost || candidate < *best_cost)
            {
                best_cost = candidate;
                best_preferred_mask = preferred_mask;
            }
        }
        ASSERT_TRUE(best_cost.has_value());
        const auto &actual =
            result.planned_plan.placements[0].routed_expert_tier;
        for (int expert = 0; expert < kExperts; ++expert)
        {
            const int expected =
                ((best_preferred_mask >> expert) & 1u) != 0 ? 0 : 1;
            EXPECT_EQ(actual[static_cast<size_t>(expert)], expected)
                << "expert=" << expert;
        }
    }

    TEST(
        Test__MoERoutedTierRebalancer,
        ServiceProfileMustBeTotalPositiveAndAllowsMeasuredPhaseCrossover)
    {
        auto plan = allGpuPlan();
        std::vector<std::vector<uint64_t>> counts(
            kLayers, std::vector<uint64_t>(kExperts, 1));
        auto histogram = makeHistogram(counts);
        auto profile = serviceProfile(
            plan,
            /* preferred */ {5, 5, 5},
            /* less preferred */ {6, 6, 6});

        MoERoutedExpertPlacementPlannerOptions options;
        options.decode_histogram = histogram.get();
        options.phase_service_profile = &profile;

        profile.costs.pop_back();
        EXPECT_THROW(
            (void)MoERoutedExpertPlacementPlanner::plan(
                plan, metadata(), options),
            std::invalid_argument);

        profile = serviceProfile(
            plan,
            /* preferred */ {5, 5, 5},
            /* less preferred */ {6, 4, 6});
        const auto crossover = MoERoutedExpertPlacementPlanner::plan(
            plan, metadata(), options);
        EXPECT_TRUE(
            crossover.rebalance_diagnostics.phase_service_profile_used)
            << "Exact measured phase economics must remain authoritative when backends cross over";

        profile = serviceProfile(
            plan,
            /* preferred */ {5, 0, 5},
            /* less preferred */ {6, 6, 6});
        EXPECT_THROW(
            (void)MoERoutedExpertPlacementPlanner::plan(
                plan, metadata(), options),
            std::invalid_argument);
    }

} // namespace llaminar2::test
