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
#include <array>
#include <map>
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
        ASSERT_EQ(proposals.size(), 2u);
        const std::array policies{
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
            DenseParallelPolicy::TensorParallel};
        for (std::size_t index = 0; index < proposals.size(); ++index)
        {
            const auto &proposal = proposals[index];
            EXPECT_TRUE(proposal.config.tp_devices.empty());
            ASSERT_TRUE(proposal.config.moe_routed_expert_plan);
            EXPECT_EQ(proposal.config.moe_routed_expert_plan
                          ->continuation_domain_spec.effectiveDensePolicy(),
                      policies[index]);
            compile(proposal, true);
            ExecutionPlanBuilder builder;
            const auto resolved = ResolvedRankOrchestration::resolve(
                proposal.config, model(true),
                proposal.membership.inventory(), builder, 0);
            ASSERT_TRUE(resolved.overlayExecution());
            ASSERT_TRUE(resolved.config().moe_routed_expert_plan);
            EXPECT_EQ(resolved.config().moe_routed_expert_plan->domains.size(), 1u);
            EXPECT_EQ(resolved.config().moe_rebalance.mode, mode);
        }
    }
}

TEST(AutomaticOrchestrationCandidates,
     MultiTierLocalGpuContinuationOffersBothAdmittedDensePolicies)
{
    auto inventory = hosts(2);
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.ranks[1].local_rank = 1;
    inventory.buildNodeAggregations();
    cards(inventory, 0, DeviceType::CUDA, 2);
    auto config = request({DeviceType::CUDA, DeviceType::CPU},
                          {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts =
        {{DeviceType::CUDA, 2}, {DeviceType::CPU, 2}};
    const auto proposals = candidates(config, inventory, true);
    ASSERT_EQ(proposals.size(), 2u);
    const std::array policies{
        DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
        DenseParallelPolicy::TensorParallel};
    for (std::size_t index = 0; index < proposals.size(); ++index)
    {
        const auto &proposal = proposals[index];
        ASSERT_TRUE(proposal.config.moe_routed_expert_plan);
        EXPECT_EQ(proposal.config.moe_routed_expert_plan
                      ->continuation_domain_spec.effectiveDensePolicy(),
                  policies[index]);
        EXPECT_EQ(proposal.membership.discoveryRanks(),
                  (std::vector<int>{0, 1}));
        compile(proposal, true);
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

TEST(AutomaticOrchestrationCandidates, ExactDeviceCountsKeepAutomaticTPAndPPChoices)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto inventory = hosts(1);
        cards(inventory, 0, backend, 4);
        auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
        config.automatic_planning.device_counts = {{backend, 2}};
        const auto tp = candidates(config, inventory);
        ASSERT_EQ(tp.size(), 6u); // All two-device subsets, not hard-coded ordinals.
        for (const auto &proposal : tp)
        {
            EXPECT_EQ(proposal.config.tp_devices.size(), 2u);
            compile(proposal);
        }
        config.automatic_planning.only_strategies = {OrchestrationStrategy::PipelineParallel};
        const auto pp = candidates(config, inventory);
        ASSERT_EQ(pp.size(), 36u); // Twelve ordered pairs, three legal layer splits.
        for (const auto &proposal : pp)
        {
            ASSERT_EQ(proposal.config.domain_definitions.size(), 2u);
            for (const auto &domain : proposal.config.domain_definitions) EXPECT_EQ(domain.devices.size(), 1u);
            compile(proposal);
        }
        config.automatic_planning.device_counts = {{backend, 5}};
        EXPECT_TRUE(candidates(config, inventory).empty());
    }
}

TEST(AutomaticOrchestrationCandidates, ExactDeviceCountsRequireBothVendorsInHybridTPPP)
{
    auto inventory = hosts(1);
    cards(inventory, 0, DeviceType::CUDA, 2);
    cards(inventory, 0, DeviceType::ROCm, 4);
    auto config = request({DeviceType::CUDA, DeviceType::ROCm}, {OrchestrationStrategy::PipelineParallel});
    config.automatic_planning.device_counts = {{DeviceType::CUDA, 2}, {DeviceType::ROCm, 2}};
    const auto proposals = candidates(config, inventory);
    ASSERT_EQ(proposals.size(), 36u); // Six ROCm subsets, both orders, three splits.
    for (const auto &proposal : proposals)
    {
        ASSERT_EQ(proposal.config.domain_definitions.size(), 2u);
        std::set<DeviceType> vendors;
        for (const auto &domain : proposal.config.domain_definitions)
        {
            ASSERT_EQ(domain.devices.size(), 2u);
            EXPECT_EQ(domain.devices[0].device_type, domain.devices[1].device_type);
            vendors.insert(domain.devices[0].device_type);
        }
        EXPECT_EQ(vendors, (std::set{DeviceType::CUDA, DeviceType::ROCm}));
        compile(proposal);
    }
}

/**
 * @brief Reproduce a two-socket host where both MPI ranks can enumerate all GPUs.
 *
 * An automatic 2xCUDA/4xROCm overlay previously selected both tiers on rank
 * zero.  That passed inventory and memory admission, then failed while the
 * CUDA graph tried to lower ROCm participant work without a rank executor.
 */
TEST(AutomaticOrchestrationCandidates,
     ExactDeviceCountsMixedVendorOverlayRequiresRealRankBoundary)
{
    auto inventory = hosts(2);
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.ranks[1].local_rank = 1;
    cards(inventory, 0, DeviceType::CUDA, 2);
    cards(inventory, 0, DeviceType::ROCm, 4);
    for (auto &gpu : inventory.ranks[0].gpus)
        if (gpu.type == DeviceType::ROCm)
            gpu.numa_node = inventory.ranks[1].cpu.numa_node;
    inventory.ranks[1].gpus = inventory.ranks[0].gpus;
    inventory.buildNodeAggregations();

    auto config = request({DeviceType::CUDA, DeviceType::ROCm},
        {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts =
        {{DeviceType::CUDA, 2}, {DeviceType::ROCm, 4}};
    const auto proposals = candidates(config, inventory, true);
    ASSERT_EQ(proposals.size(), 8u); // Both continuation directions and dense policies.
    for (const auto &proposal : proposals)
    {
        EXPECT_EQ(proposal.membership.discoveryRanks().size(), 2u);
        const auto &domains = proposal.config.moe_routed_expert_plan->domains;
        ASSERT_EQ(domains.size(), 2u);
        EXPECT_NE(domains[0].owner_rank, domains[1].owner_rank);
        compile(proposal, true);
    }
}

/**
 * @brief A public auto request can require every participant in three tiers.
 *
 * Both MPI ranks see the same GPUs, but only the rank on the other socket can
 * execute the foreign GPU tier. CPU remains a two-socket domain rather than
 * two idle one-socket decorations. Search may also propose finer GPU-tier
 * partitions; every published candidate still compiles and keeps the exact
 * requested physical counts.
 */
TEST(AutomaticOrchestrationCandidates,
     MultiTierOverlayUsesAllCUDAAndROCmAndCPUParticipants)
{
    auto inventory = hosts(2);
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.ranks[1].local_rank = 1;
    cards(inventory, 0, DeviceType::CUDA, 2);
    cards(inventory, 0, DeviceType::ROCm, 4);
    for (auto &gpu : inventory.ranks[0].gpus)
        if (gpu.type == DeviceType::ROCm)
            gpu.numa_node = inventory.ranks[1].cpu.numa_node;
    inventory.ranks[1].gpus = inventory.ranks[0].gpus;
    inventory.buildNodeAggregations();

    auto config = request({DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm},
        {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts = {
        {DeviceType::CPU, 2}, {DeviceType::CUDA, 2}, {DeviceType::ROCm, 4}};
    const auto proposals = candidates(config, inventory, true);
    ASSERT_FALSE(proposals.empty());
    bool three_tier_cuda_continuation = false;
    for (const auto &proposal : proposals)
    {
        const auto &overlay = *proposal.config.moe_routed_expert_plan;
        ASSERT_GE(overlay.domains.size(), 3u);
        ASSERT_EQ(overlay.routed_tiers.size(), overlay.domains.size());
        EXPECT_EQ(proposal.membership.discoveryRanks().size(), 2u);
        std::map<DeviceType, std::size_t> participant_counts;
        for (std::size_t index = 0; index < overlay.domains.size(); ++index)
        {
            const auto &domain = overlay.domains[index];
            EXPECT_EQ(overlay.routed_tiers[index].domain, domain.name);
            EXPECT_EQ(overlay.routed_tiers[index].priority,
                      static_cast<int>(index));
            for (const auto &participant : domain.participants)
                ++participant_counts[participant.device_type];
        }
        EXPECT_EQ(participant_counts[DeviceType::CUDA], 2u);
        EXPECT_EQ(participant_counts[DeviceType::ROCm], 4u);
        EXPECT_EQ(participant_counts[DeviceType::CPU], 2u);
        three_tier_cuda_continuation |=
            overlay.domains.size() == 3u &&
            overlay.domains.front().participants.front().isCUDA();
        compile(proposal, true);
    }
    EXPECT_TRUE(three_tier_cuda_continuation);
}

/**
 * @brief Search depth follows disjoint physical ownership, not a tier limit.
 *
 * A second CUDA owner rank cannot be folded into the first rank-local GPU
 * domain. The planner must therefore represent two CUDA domains, one ROCm
 * domain, and a node-wide CPU domain in the same candidate.
 */
TEST(AutomaticOrchestrationCandidates,
     MultiTierOverlayCanExtendBeyondThreeDomains)
{
    auto inventory = hosts(3);
    for (int rank = 1; rank < 3; ++rank)
    {
        inventory.ranks[rank].node_id = inventory.ranks[0].node_id;
        inventory.ranks[rank].hostname = inventory.ranks[0].hostname;
        inventory.ranks[rank].local_rank = rank;
    }
    cards(inventory, 0, DeviceType::CUDA, 1);
    cards(inventory, 1, DeviceType::ROCm, 1);
    cards(inventory, 2, DeviceType::CUDA, 1);
    inventory.ranks[2].gpus.front().local_device_id = 5;
    inventory.ranks[2].gpus.front().uuid = "cuda-other-owner";
    inventory.buildNodeAggregations();

    auto config = request({DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm},
        {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts = {
        {DeviceType::CPU, 3}, {DeviceType::CUDA, 2}, {DeviceType::ROCm, 1}};
    const auto proposals = candidates(config, inventory, true);
    ASSERT_FALSE(proposals.empty());
    bool four_domains = false;
    for (const auto &proposal : proposals)
    {
        const auto &overlay = *proposal.config.moe_routed_expert_plan;
        if (overlay.domains.size() != 4u) continue;
        four_domains = true;
        ASSERT_EQ(overlay.routed_tiers.size(), 4u);
        for (std::size_t index = 0; index < overlay.routed_tiers.size(); ++index)
            EXPECT_EQ(overlay.routed_tiers[index].priority,
                      static_cast<int>(index));
        compile(proposal, true);
    }
    EXPECT_TRUE(four_domains);

    // Omission of cardinality is still genuine auto search: it must not
    // silently restore the historical pair-only domain ceiling.
    config.automatic_planning.device_counts.reset();
    const auto unconstrained = candidates(config, inventory, true);
    EXPECT_TRUE(std::any_of(unconstrained.begin(), unconstrained.end(),
        [](const AutomaticOrchestrationCandidate &proposal)
        {
            return proposal.config.moe_routed_expert_plan &&
                   proposal.config.moe_routed_expert_plan->domains.size() >= 4u;
        }));
}

/**
 * @brief Adding another independently owned GPU pool adds another tier.
 *
 * The mixed-vendor planner must not replace the old pair-only ceiling with a
 * new fixed count. Each GPU pool below belongs to a distinct rank; the CPU
 * domain spans all four NUMA owners on the same physical node.
 */
TEST(AutomaticOrchestrationCandidates,
     MultiTierOverlayCanExtendToFiveDomains)
{
    auto inventory = hosts(4);
    for (int rank = 1; rank < 4; ++rank)
    {
        inventory.ranks[rank].node_id = inventory.ranks[0].node_id;
        inventory.ranks[rank].hostname = inventory.ranks[0].hostname;
        inventory.ranks[rank].local_rank = rank;
    }
    cards(inventory, 0, DeviceType::CUDA, 1);
    cards(inventory, 1, DeviceType::ROCm, 1);
    cards(inventory, 2, DeviceType::CUDA, 1);
    cards(inventory, 3, DeviceType::ROCm, 1);
    inventory.ranks[2].gpus.front().local_device_id = 5;
    inventory.ranks[2].gpus.front().uuid = "cuda-other-owner";
    inventory.ranks[3].gpus.front().local_device_id = 5;
    inventory.ranks[3].gpus.front().uuid = "rocm-other-owner";
    inventory.buildNodeAggregations();

    auto config = request({DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm},
        {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts = {
        {DeviceType::CPU, 4}, {DeviceType::CUDA, 2}, {DeviceType::ROCm, 2}};
    const auto proposals = candidates(config, inventory, true);
    const auto found = std::find_if(proposals.begin(), proposals.end(),
        [](const AutomaticOrchestrationCandidate &proposal)
        {
            const auto &overlay = proposal.config.moe_routed_expert_plan;
            return overlay && overlay->domains.size() == 5u &&
                   overlay->domains.front().participants.front().isCUDA();
        });
    ASSERT_NE(found, proposals.end());
    ASSERT_EQ(found->config.moe_routed_expert_plan->routed_tiers.size(), 5u);
    compile(*found, true);
}

TEST(AutomaticOrchestrationCandidates, ExactDeviceCountsUsePhysicalNotRankCardinality)
{
    auto inventory = hosts(2);
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    cards(inventory, 0, DeviceType::ROCm, 1);
    inventory.ranks[1].gpus = inventory.ranks[0].gpus;
    inventory.buildNodeAggregations();
    auto config = request({DeviceType::ROCm}, {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts = {{DeviceType::ROCm, 2}};
    EXPECT_TRUE(candidates(config, inventory, true).empty());

    // A three-host CPU pool loses the continuation host during remote offload
    // construction. That removed socket cannot satisfy a requested CPU count.
    inventory = hosts(3);
    cards(inventory, 0, DeviceType::CUDA, 1);
    config = request({DeviceType::CUDA, DeviceType::CPU}, {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts = {{DeviceType::CUDA, 1}, {DeviceType::CPU, 3}};
    EXPECT_TRUE(candidates(config, inventory, true).empty());
    config.automatic_planning.device_counts = {{DeviceType::CUDA, 1}, {DeviceType::CPU, 2}};
    const auto proposals = candidates(config, inventory, true);
    ASSERT_EQ(proposals.size(), 1u);
    EXPECT_EQ(proposals.front().config.moe_routed_expert_plan->domains[1].participants.size(), 2u);
}

/**
 * @brief Default MoE auto search never publishes an authority-less PP plan.
 *
 * The default policy admits every implemented strategy.  A dense pipeline
 * proposal does not encode routed-expert domains or continuation ownership,
 * so treating it as a MoE candidate made an otherwise valid local GPU search
 * fail while compiling an unrelated proposal.  Every emitted routed-model
 * candidate must instead compile with the sole typed ExpertOverlay authority.
 */
TEST(AutomaticOrchestrationCandidates,
     DefaultMoESearchExcludesAuthoritylessPipelineCandidates)
{
    auto inventory = hosts(1);
    cards(inventory, 0, DeviceType::ROCm, 2);
    auto config = request(
        {DeviceType::ROCm},
        {OrchestrationStrategy::SingleDevice,
         OrchestrationStrategy::TensorParallel,
         OrchestrationStrategy::PipelineParallel,
         OrchestrationStrategy::ExpertOverlay});

    const auto proposals = candidates(config, inventory, true);
    ASSERT_FALSE(proposals.empty());
    for (const auto &proposal : proposals)
    {
        EXPECT_NE(proposal.strategy, OrchestrationStrategy::PipelineParallel);
        if (proposal.strategy != OrchestrationStrategy::SingleDevice)
        {
            ASSERT_TRUE(proposal.config.moe_routed_expert_plan);
            EXPECT_TRUE(
                proposal.config.moe_routed_expert_plan
                    ->usesExpertOverlayAuthority());
        }
        compile(proposal, true);
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
