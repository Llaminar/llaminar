/**
 * @file Test__AutomaticOrchestrationCandidates.cpp
 * @brief Device-free coverage of automatic topology construction and compilation.
 *
 * Synthetic observations deliberately separate MPI rank, node, NUMA and GPU
 * ordinal. Proposals are checked through the real model-aware rank compiler and
 * reusable config codec. These tests establish placement semantics only; no
 * synthetic memory number or observed core count is treated as cost evidence.
 */
#include "planning/AutomaticOrchestrationCandidates.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/ResolvedRankOrchestration.h"
#include "app/RuntimeInitPhase.h"
#include "config/OrchestrationConfigDocument.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "utils/NUMATopology.h"
#include <gtest/gtest.h>
#include <set>

using namespace llaminar2;

namespace
{
    /** @return Exact main/MTP boundary without loading model tensors. */
    PlanningModelMetadata model(bool moe = false)
    {
        ModelMemoryProfile profile;
        profile.n_layers = 5;
        profile.d_model = 512;
        profile.d_ff = 1536;
        profile.n_heads = 32;
        profile.n_kv_heads = 2;
        profile.head_dim = 16;
        profile.vocab_size = 320;
        profile.max_seq_len = 8192;
        profile.expert_count = moe ? 32 : 0;
        return PlanningModelMetadata(std::move(profile), 4);
    }

    /** @return Distinct physical hosts whose NUMA IDs are not communicator ranks. */
    ClusterInventory hosts(int count)
    {
        ClusterInventory inventory;
        inventory.world_size = count;
        for (int rank = 0; rank < count; ++rank)
        {
            RankInventory observed;
            observed.rank = rank;
            observed.node_id = 10 + rank;
            observed.local_rank = 0;
            observed.hostname = "host-" + std::to_string(rank);
            observed.cpu.numa_node = 9 - rank * 2;
            observed.cpu_cores = 8;
            observed.cpu_worker_threads = 8;
            observed.numa_nodes = 1;
            inventory.ranks.push_back(std::move(observed));
        }
        inventory.buildNodeAggregations();
        return inventory;
    }

    /** @brief Add observed GPUs with sparse ordinals, never initialize a driver. */
    void cards(ClusterInventory &inventory, int rank, DeviceType backend, int count)
    {
        for (int index = 0; index < count; ++index)
        {
            DeviceInfo gpu;
            gpu.type = backend;
            gpu.local_device_id = index * 2 + 3;
            gpu.numa_node = inventory.ranks[rank].cpu.numa_node;
            gpu.uuid = std::string(deviceTypeToString(backend)) + "-" + std::to_string(index);
            gpu.compute_units = 64;
            inventory.ranks[rank].gpus.push_back(std::move(gpu));
        }
        inventory.buildNodeAggregations();
    }

    /** @return A policy-only automatic request with exact hard restrictions. */
    OrchestrationConfig request(std::vector<DeviceType> backends,
                                std::vector<OrchestrationStrategy> strategies)
    {
        OrchestrationConfig config;
        config.automatic_planning.only_backends = std::move(backends);
        config.automatic_planning.only_strategies = std::move(strategies);
        return config;
    }

    /** @return Small test-only materialization of the production streaming visitor. */
    std::vector<AutomaticOrchestrationCandidate> candidates(const OrchestrationConfig &config,
        const ClusterInventory &inventory, bool moe = false)
    {
        std::vector<AutomaticOrchestrationCandidate> result;
        visitAutomaticOrchestrationCandidates(config, model(moe), inventory,
            [&](auto candidate) { result.push_back(std::move(candidate)); });
        return result;
    }

    /** @brief Compile every selected rank through the installed production boundary. */
    void compile(const AutomaticOrchestrationCandidate &candidate, bool moe = false)
    {
        ExecutionPlanBuilder builder;
        const auto document = serializeOrchestrationConfig(candidate.config);
        const auto applied = deserializeOrchestrationConfig(document);
        EXPECT_EQ(serializeOrchestrationConfig(applied), document);
        for (int rank = 0; rank < candidate.membership.inventory().world_size; ++rank)
        {
            const auto resolved = ResolvedRankOrchestration::resolve(candidate.config, model(moe),
                candidate.membership.inventory(), builder, rank);
            const auto from_document = ResolvedRankOrchestration::resolve(applied, model(moe),
                candidate.membership.inventory(), builder, rank);
            EXPECT_EQ(from_document.rankPlan().toString(), resolved.rankPlan().toString());
            EXPECT_EQ(resolved.rankPlan().rank, rank);
            EXPECT_TRUE(resolved.rankPlan().validate().empty());
        }
    }
}

TEST(AutomaticOrchestrationCandidates, SingleDeviceCanSelectANonzeroDiscoveryRankOnEveryBackend)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        auto inventory = hosts(3);
        if (backend != DeviceType::CPU) cards(inventory, 2, backend, 1);
        const auto config = request({backend}, {OrchestrationStrategy::SingleDevice});
        const auto proposals = candidates(config, inventory);
        ASSERT_EQ(proposals.size(), backend == DeviceType::CPU ? 3u : 1u);
        const auto &chosen = proposals.back();
        EXPECT_EQ(chosen.membership.discoveryRanks(), std::vector<int>({2}));
        EXPECT_EQ(chosen.config.planning_mode, OrchestrationPlanningMode::Apply);
        EXPECT_FALSE(chosen.config.automatic_planning.specified());
        ASSERT_TRUE(chosen.config.device_for_this_rank);
        EXPECT_EQ(chosen.config.device_for_this_rank->numa_node, 5);
        EXPECT_EQ(chosen.config.device_for_this_rank->device_type, backend);
        EXPECT_EQ(chosen.config.device_for_this_rank_numa_explicit, backend == DeviceType::CPU);
        if (backend == DeviceType::CPU)
        {
            // Apply the serialized proposal after rank compaction: execution
            // rank zero still belongs to the observed CPU on NUMA node five.
            const auto applied = deserializeOrchestrationConfig(serializeOrchestrationConfig(chosen.config));
            const NUMAInfo observed{.local_numa_node = 5, .total_numa_nodes = 10,
                .detection_succeeded = true, .detection_method = "fixture"};
            EXPECT_EQ(RuntimeInitPhase::resolveCPUBackendNUMANode(applied, 0, 1, observed), 5);
        }
        EXPECT_TRUE(std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(chosen.config)));
        compile(chosen);
        EXPECT_FALSE(config.device_for_this_rank);
    }
}

TEST(AutomaticOrchestrationCandidates, GPUChoicesIncludeEverySubsetThroughEightParticipants)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto inventory = hosts(1);
        cards(inventory, 0, backend, 8);
        const auto config = request({backend}, {OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel});
        const auto proposals = candidates(config, inventory);
        EXPECT_EQ(proposals.size(), 255u);
        std::set<std::vector<int>> subsets;
        for (const auto &proposal : proposals)
        {
            std::vector<int> ordinals;
            if (proposal.config.device_for_this_rank)
                ordinals.push_back(proposal.config.device_for_this_rank->device_ordinal);
            else for (const auto &device : proposal.config.tp_devices)
                ordinals.push_back(device.device_ordinal);
            EXPECT_TRUE(subsets.insert(std::move(ordinals)).second);
        }
        // Compiler validation, including head divisibility, is a separate
        // admission step. Exercise legal TP degrees; enumeration must not
        // silently use fake model geometry to admit an illegal degree.
        for (const auto &proposal : proposals)
            if (!proposal.config.tp_devices.empty())
            {
                const auto size = proposal.config.tp_devices.size();
                if (size == 2 || size == 4 || size == 8) compile(proposal);
            }
    }
}

TEST(AutomaticOrchestrationCandidates, LocalMoETPCompilesWithOneOverlayAuthority)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Dynamic})
    {
        SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend) << " mode=" << static_cast<int>(mode));
        auto inventory = hosts(1);
        if (backend == DeviceType::CPU)
        {
            auto &rank = inventory.ranks.front();
            rank.cpu.numa_node = -1;
            rank.numa_nodes = 2;
            rank.cpu_socket_info = {{.numa_node = 3}, {.numa_node = 7}};
        }
        else cards(inventory, 0, backend, 2);
        auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
        config.moe_rebalance.mode = mode;
        const auto proposals = candidates(config, inventory, true);
        ASSERT_EQ(proposals.size(), 1u);
        ASSERT_EQ(proposals.front().config.tp_devices.size(), 2u);
        EXPECT_TRUE(proposals.front().config.domain_definitions.empty());
        compile(proposals.front(), true);
        ExecutionPlanBuilder builder;
        const auto resolved = ResolvedRankOrchestration::resolve(proposals.front().config, model(true),
            proposals.front().membership.inventory(), builder, 0);
        ASSERT_TRUE(resolved.overlayExecution());
        ASSERT_TRUE(resolved.config().moe_routed_expert_plan);
        EXPECT_EQ(resolved.config().moe_routed_expert_plan->domains.size(), 1u);
        EXPECT_EQ(resolved.config().moe_rebalance.mode, mode);
    }
}

TEST(AutomaticOrchestrationCandidates, CrossRankMoETPRetainsOneExplicitAuthorityAndPhysicalScope)
{
    for (bool same_host : {false, true})
        for (auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Dynamic})
        {
            auto inventory = hosts(2);
            if (same_host)
            {
                inventory.ranks[1].node_id = inventory.ranks[0].node_id;
                inventory.ranks[1].hostname = inventory.ranks[0].hostname;
                inventory.ranks[1].local_rank = 1;
                inventory.buildNodeAggregations();
            }
            auto config = request({DeviceType::CPU}, {OrchestrationStrategy::TensorParallel});
            config.moe_rebalance.mode = mode;
            const auto proposals = candidates(config, inventory, true);
            ASSERT_EQ(proposals.size(), 1u);
            const auto &proposal = proposals.front();
            EXPECT_EQ(proposal.strategy, OrchestrationStrategy::TensorParallel);
            ASSERT_TRUE(proposal.config.moe_routed_expert_plan);
            const auto &plan = *proposal.config.moe_routed_expert_plan;
            ASSERT_EQ(plan.domains.size(), 1u);
            ASSERT_EQ(plan.routed_tiers.size(), 1u);
            EXPECT_EQ(plan.domains.front().world_ranks, (std::vector<int>{0, 1}));
            EXPECT_EQ(plan.domains.front().scope, same_host ? ExecutionDomainScope::NODE_LOCAL : ExecutionDomainScope::GLOBAL);
            EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
            EXPECT_TRUE(proposal.config.pp_stage_definitions.empty());
            EXPECT_EQ(proposal.config.moe_rebalance.mode, mode);
            compile(proposal, true);
        }
}

TEST(AutomaticOrchestrationCandidates, PipelinesTryEveryRealMainLayerBoundaryInBothDirections)
{
    const auto inventory = hosts(2);
    const auto proposals = candidates(request({DeviceType::CPU}, {OrchestrationStrategy::PipelineParallel}), inventory);
    ASSERT_EQ(proposals.size(), 6u);
    std::set<std::pair<int, int>> splits;
    for (const auto &proposal : proposals)
    {
        ASSERT_EQ(proposal.config.pp_stage_definitions.size(), 2u);
        const auto &first = proposal.config.pp_stage_definitions[0];
        const auto &last = proposal.config.pp_stage_definitions[1];
        EXPECT_EQ(first.first_layer, 0);
        EXPECT_EQ(last.first_layer, first.last_layer + 1);
        EXPECT_EQ(last.last_layer, 3) << "The MTP block cannot enter the main pipeline";
        EXPECT_TRUE(splits.emplace(proposal.membership.discoveryRanks().front(), last.first_layer).second);
        compile(proposal);
    }
}

TEST(AutomaticOrchestrationCandidates, RemoteCPUOverlaysRebaseEveryOwnerAndPreserveServingPolicy)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (int remote_count : {1, 2})
        {
            SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend) << " remote=" << remote_count);
            auto inventory = hosts(remote_count + 1);
            cards(inventory, remote_count, backend, 1);
            auto config = request({backend, DeviceType::CPU}, {OrchestrationStrategy::ExpertOverlay});
            config.mtp.enabled = true;
            config.mtp.graph_capacity_draft_tokens = 15;
            config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
            config.routed_expert_owner_order = RoutedExpertOwnerOrder::Random;
            config.prefix_cache.enabled = true;
            config.max_seq_len = 8192;
            config.prefill_max_bucket_size = 512;
            const auto proposals = candidates(config, inventory, true);
            int witnesses = 0;
            for (const auto &proposal : proposals)
            {
                const auto &overlay = *proposal.config.moe_routed_expert_plan;
                const auto &cpu = overlay.domains[1];
                if (cpu.participants.size() != static_cast<size_t>(remote_count) ||
                    std::any_of(cpu.participants.begin(), cpu.participants.end(), [&](const auto &device) {
                        return device.hostname == inventory.ranks[remote_count].hostname;
                    })) continue;
                ++witnesses;
                EXPECT_EQ(proposal.membership.discoveryRanks().front(), remote_count);
                for (int index = 0; index < remote_count; ++index)
                {
                    EXPECT_EQ(cpu.world_ranks[index], index + 1);
                    EXPECT_EQ(cpu.participants[index].hostname, inventory.ranks[index].hostname);
                    EXPECT_EQ(cpu.participants[index].numa_node, inventory.ranks[index].cpu.numa_node);
                }
                EXPECT_EQ(cpu.scope, remote_count == 1 ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::GLOBAL);
                EXPECT_EQ(overlay.owner_order, RoutedExpertOwnerOrder::Random);
                EXPECT_EQ(overlay.residency_policy, RoutedExpertResidencyPolicy::RoutedTierRebalanced);
                for (const auto &tier : overlay.routed_tiers)
                {
                    EXPECT_EQ(tier.max_experts_per_layer, 0);
                    EXPECT_EQ(tier.memory_budget_bytes, 0u);
                    EXPECT_TRUE(tier.resolved_live_experts_per_layer.empty());
                }
                EXPECT_TRUE(proposal.config.mtp.enabled);
                EXPECT_EQ(proposal.config.mtp.graph_capacity_draft_tokens, 15);
                EXPECT_TRUE(proposal.config.prefix_cache.enabled);
                EXPECT_EQ(proposal.config.prefill_max_bucket_size, 512);
                compile(proposal, true);
            }
            EXPECT_EQ(witnesses, 1);
            EXPECT_FALSE(config.moe_routed_expert_plan);
            EXPECT_TRUE(config.domain_definitions.empty());
        }
}

TEST(AutomaticOrchestrationCandidates, DenseModelsNeverAdvertiseExpertOverlay)
{
    auto inventory = hosts(2);
    cards(inventory, 0, DeviceType::CUDA, 1);
    EXPECT_TRUE(candidates(request({DeviceType::CPU, DeviceType::CUDA},
        {OrchestrationStrategy::ExpertOverlay}), inventory, false).empty());
}

TEST(AutomaticOrchestrationCandidates, AllHostsRequiresEveryPhysicalHostNotEveryRank)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (int remotes : {1, 2})
    {
        SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend) << " remote_hosts=" << remotes);
        auto inventory = hosts(remotes + 1);
        cards(inventory, 0, backend, 1);
        auto config = request({DeviceType::CPU, backend}, {OrchestrationStrategy::ExpertOverlay});
        const auto unrestricted = candidates(config, inventory, true);
        config.automatic_planning.host_participation = AutomaticHostParticipation::AllDiscovered;
        const auto complete = candidates(config, inventory, true);
        ASSERT_FALSE(complete.empty());
        EXPECT_LT(complete.size(), unrestricted.size());
        for (const auto &proposal : complete)
        {
            std::set<int> nodes;
            for (const int rank : proposal.membership.discoveryRanks()) nodes.insert(inventory.ranks[rank].node_id);
            EXPECT_EQ(nodes.size(), remotes + 1);
            EXPECT_FALSE(proposal.config.automatic_planning.specified());
            compile(proposal, true);
        }
    }
    auto inventory = hosts(2);
    // Two process owners on one physical host must not force a two-rank plan.
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.buildNodeAggregations();
    auto config = request({DeviceType::CPU}, {OrchestrationStrategy::SingleDevice});
    config.automatic_planning.host_participation = AutomaticHostParticipation::AllDiscovered;
    const auto same_host = candidates(config, inventory);
    ASSERT_FALSE(same_host.empty());
    for (const auto &proposal : same_host) EXPECT_EQ(proposal.membership.discoveryRanks().size(), 1u);
    // Conversely, an impossible single-device/all-host request stays empty;
    // the planner must not silently relax host participation to make it fit.
    EXPECT_TRUE(candidates(config, hosts(2)).empty());
}

TEST(AutomaticOrchestrationCandidates, DuplicateGPUVisibilityCannotBecomeTwoExpertTiers)
{
    auto inventory = hosts(2);
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    cards(inventory, 0, DeviceType::ROCm, 1);
    inventory.ranks[1].gpus = inventory.ranks[0].gpus;
    inventory.buildNodeAggregations();
    EXPECT_TRUE(candidates(request({DeviceType::ROCm}, {OrchestrationStrategy::ExpertOverlay}), inventory, true).empty());
}

TEST(AutomaticOrchestrationCandidates, MixedVendorTiersConsiderBothContinuationDirections)
{
    auto inventory = hosts(2);
    cards(inventory, 0, DeviceType::CUDA, 1);
    cards(inventory, 1, DeviceType::ROCm, 1);
    const auto proposals = candidates(request({DeviceType::CUDA, DeviceType::ROCm},
        {OrchestrationStrategy::ExpertOverlay}), inventory, true);
    ASSERT_EQ(proposals.size(), 2u);
    std::set<DeviceType> continuation_backends;
    for (const auto &proposal : proposals)
    {
        const auto &overlay = *proposal.config.moe_routed_expert_plan;
        continuation_backends.insert(overlay.domains.front().participants.front().device_type);
        EXPECT_EQ(overlay.routed_tiers[0].priority, 0);
        EXPECT_EQ(overlay.routed_tiers[1].priority, 1);
        compile(proposal, true);
    }
    EXPECT_EQ(continuation_backends, std::set<DeviceType>({DeviceType::CUDA, DeviceType::ROCm}));
}

TEST(AutomaticOrchestrationCandidates, SoftHintsNeitherRemoveCandidatesNorBecomeAFirstFitOrder)
{
    auto inventory = hosts(1);
    cards(inventory, 0, DeviceType::CUDA, 1);
    cards(inventory, 0, DeviceType::ROCm, 1);
    auto config = request({DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm},
        {OrchestrationStrategy::SingleDevice});
    const auto plain = candidates(config, inventory);
    config.automatic_planning.prefer_backend = DeviceType::ROCm;
    config.automatic_planning.prefer_strategy = OrchestrationStrategy::SingleDevice;
    const auto hinted = candidates(config, inventory);
    ASSERT_EQ(plain.size(), 3u);
    ASSERT_EQ(hinted.size(), plain.size());
    for (size_t index = 0; index < plain.size(); ++index)
        EXPECT_EQ(serializeOrchestrationConfig(hinted[index].config), serializeOrchestrationConfig(plain[index].config));
}

TEST(AutomaticOrchestrationCandidates, VisitorFailureStopsConstructionAndIsNotAnEmptySuccessfulSearch)
{
    const auto inventory = hosts(2);
    const auto config = request({DeviceType::CPU}, {OrchestrationStrategy::SingleDevice});
    int calls = 0;
    EXPECT_THROW(visitAutomaticOrchestrationCandidates(config, model(), inventory, [&](auto) {
        ++calls;
        throw std::runtime_error("candidate admission failure");
    }), std::runtime_error);
    EXPECT_EQ(calls, 1);
}

TEST(AutomaticOrchestrationCandidates, InvalidIntentOrObservationFailsBeforePublishingCandidates)
{
    auto inventory = hosts(1);
    auto config = request({DeviceType::CPU}, {OrchestrationStrategy::SingleDevice});
    config.device_for_this_rank = GlobalDeviceAddress::cpu(9, "host-0");
    EXPECT_THROW(candidates(config, inventory), std::invalid_argument);
    config.device_for_this_rank.reset();
    inventory.ranks[0].cpu.numa_node = -1;
    EXPECT_THROW(candidates(config, inventory), std::invalid_argument);
    inventory = hosts(1);
    inventory.ranks[0].rank = 4;
    EXPECT_THROW(candidates(config, inventory), std::invalid_argument);
    inventory = hosts(1);
    cards(inventory, 0, DeviceType::CUDA, 1);
    inventory.ranks[0].gpus[0].uuid.clear();
    EXPECT_THROW(candidates(request({DeviceType::CUDA}, {OrchestrationStrategy::SingleDevice}), inventory), std::invalid_argument);
}
