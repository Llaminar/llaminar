/**
 * @file Test__AutomaticOrchestrationCandidates.cpp
 * @brief Device-free coverage of automatic topology construction and compilation.
 *
 * Synthetic observations deliberately separate MPI rank, node, NUMA and GPU
 * ordinal. Proposals are checked through the real model-aware rank compiler and
 * reusable config codec. These tests establish placement semantics only; no
 * synthetic memory number or observed core count is treated as cost evidence.
 *
 * Original QFN declarations and actual nonordinal endpoint fixtures cover every
 * supported head degree without loading model tensors or initializing GPUs.
 * Automatic PP keeps stage-owned MoE authorities and exact positive stage-width
 * constraints, independently of physical NUMA IDs or local GPU cardinality.
 */
#include "planning/AutomaticOrchestrationCandidates.h"
#include "loaders/ModelContext.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/TensorParallelSourceDeclaration.h"
#include "planning/WeightShardGeometry.h"
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "planning/MoEOverlayPlanningInputs.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "planning/ResolvedRankOrchestration.h"
#include "app/RuntimeInitPhase.h"
#include "config/OrchestrationConfigDocument.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/moe/MoEProjectionArenaGeometry.h"
#include "execution/moe/MoEIntermediateShardArenaGeometry.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/moe/MoEOverlayPhysicalLaneDemand.h"
#include "utils/NUMATopology.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../models/qwen4exp/Qwen4ExpSidecarFixture.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include "../../fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include <gtest/gtest.h>
#include <array>
#include <map>
#include <numeric>
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
        profile.mtp_layer_count = 1;
        test::PlanningGGUFFixture file(moe, true);
        const auto fixture_metadata = readPlanningModelMetadata(file.path());
        for (auto tensor : fixture_metadata.memoryProfile().tensors)
            if (tensor.name.starts_with("blk.2."))
            {
                tensor.name.replace(0, 6, "blk.4.");
                tensor.layer_index = 4;
                profile.tensors.push_back(std::move(tensor));
            }
        return PlanningModelMetadata(std::move(profile), 4);
    }

    /**
     * @brief Declare this metadata fixture's complete observed no-peer topology.
     * @param original Authored rank inventory; no native runtime is queried.
     * @details Sparse ordinals retain discovery order. Every directed edge is
     * explicitly unavailable; missing observations must remain production errors.
     */
    void declareNoPeerTopology(ClusterInventory &original)
    {
        for (auto &rank : original.ranks)
            for (const auto family : {DeviceType::CUDA, DeviceType::ROCm}) {
                const auto count = std::count_if(rank.gpus.begin(), rank.gpus.end(),
                    [family](const auto &gpu) { return gpu.type == family; });
                auto &matrix = family == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
                auto &degree = family == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
                degree = static_cast<int>(count);
                matrix.assign(static_cast<std::size_t>(count * count), false);
            }
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
            // This helper creates distinct physical cards. Shared observations
            // explicitly copy a card later; never publish conflicting locality
            // before a fixture has a chance to rename a newly created device.
            gpu.uuid = std::string(deviceTypeToString(backend)) + "-" +
                std::to_string(rank) + "-" + std::to_string(index);
            gpu.compute_units = 64;
            inventory.ranks[rank].gpus.push_back(std::move(gpu));
        }
        declareNoPeerTopology(inventory);
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

/** @test Impossible native output partitions never enter pricing as valid proposals. */
TEST(AutomaticOrchestrationCandidates, ProjectionCandidatesPreserveIntegralSourceOutputPartitions)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int columns : {512, 1536})
    {
        auto inventory = hosts(1);
        cards(inventory, 0, backend, 8);
        auto profile = model(true).memoryProfile();
        profile.d_model = columns;
        profile.n_heads = columns / profile.head_dim;
        const PlanningModelMetadata source(std::move(profile), 4);
        auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
        config.routed_expert_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        std::set<std::size_t> degrees;
        visitAutomaticOrchestrationCandidates(config, source, inventory, [&](const auto &proposal) {
            ASSERT_TRUE(proposal.config.moe_routed_expert_plan);
            const auto &placement = *proposal.config.moe_routed_expert_plan;
            ASSERT_EQ(placement.domains.size(), 1u);
            const auto declaration = placement.domains.front().toExecutionDomainDefinition();
            const auto degree = declaration.participants.size();
            EXPECT_TRUE(MoEProjectionArenaGeometry::hasIntegralOutputPartition(columns, degree));
            EXPECT_EQ(declaration.resolveRoutedComputePolicy(placement.routed_tiers.size()),
                RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            EXPECT_EQ(proposal.config.routed_expert_compute_policy, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            degrees.insert(degree);
        });
        const std::set<std::size_t> expected = columns == 512
            ? std::set<std::size_t>{2, 4, 8} : std::set<std::size_t>{2, 3, 4, 6, 8};
        EXPECT_EQ(degrees, expected);
    }
}

/** @test Geometry rejection does not replace a requested mode or ban that device degree elsewhere. */
TEST(AutomaticOrchestrationCandidates, ProjectionGeometryRejectionPreservesExplicitAndMultiTierIntent)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto inventory = hosts(2);
        cards(inventory, 0, backend, 3);
        auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
        config.automatic_planning.device_counts = {{backend, 3}};
        config.routed_expert_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        EXPECT_TRUE(candidates(config, inventory, true).empty());
        config.routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
        const auto controls = candidates(config, inventory, true);
        ASSERT_EQ(controls.size(), 2u);
        for (const auto &control : controls)
            EXPECT_EQ(control.config.moe_routed_expert_plan->domains.front().routed_compute_policy,
                RoutedExpertComputePolicy::Apportioned);

        auto tiered = request({backend, DeviceType::CPU}, {OrchestrationStrategy::ExpertOverlay});
        tiered.automatic_planning.device_counts = {{backend, 3}, {DeviceType::CPU, 1}};
        const auto overlays = candidates(tiered, inventory, true);
        ASSERT_FALSE(overlays.empty());
        for (const auto &overlay : overlays)
        {
            const auto &placement = *overlay.config.moe_routed_expert_plan;
            ASSERT_GE(placement.routed_tiers.size(), 2u);
            for (const auto &domain : placement.domains)
                EXPECT_EQ(domain.toExecutionDomainDefinition().resolveRoutedComputePolicy(
                    placement.routed_tiers.size()), RoutedExpertComputePolicy::Apportioned);
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
        config.routed_expert_compute_policy = backend == DeviceType::CPU
            ? RoutedExpertComputePolicy::Apportioned : RoutedExpertComputePolicy::GateUpOwnedDownColumns;
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
            const auto expected_compute = backend == DeviceType::CPU
                ? RoutedExpertComputePolicy::Apportioned
                : RoutedExpertComputePolicy::GateUpOwnedDownColumns;
            EXPECT_EQ(resolved.config().moe_routed_expert_plan->domains.front()
                .routed_compute_policy, expected_compute);
            EXPECT_EQ(resolved.rankPlan().runtime.routed_expert_compute_policy, expected_compute);
        }
    }
}

/**
 * @brief Auto preserves the requested routed arithmetic/ownership mode through apply.
 *
 * The planner remains free to select sparse endpoint ordinals and dense work
 * policy. Neither normalization, serialization nor compilation may silently
 * turn projection ownership or replication into whole-expert apportionment.
 */
TEST(AutomaticOrchestrationCandidates, LocalMoETPPreservesRoutedComputeConstraint)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto policy : {RoutedExpertComputePolicy::Apportioned,
                                  RoutedExpertComputePolicy::Replicated,
                                  RoutedExpertComputePolicy::GateUpOwnedDownColumns})
        {
            SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend)
                << " policy=" << routedExpertComputePolicyToString(policy));
            auto inventory = hosts(1);
            cards(inventory, 0, backend, 2);
            auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
            config.routed_expert_compute_policy = policy;
            const auto proposals = candidates(config, inventory, true);
            ASSERT_EQ(proposals.size(), 2u);
            for (const auto &proposal : proposals)
            {
                compile(proposal, true);
                const auto applied = deserializeOrchestrationConfig(
                    serializeOrchestrationConfig(proposal.config));
                ExecutionPlanBuilder builder;
                const auto resolved = ResolvedRankOrchestration::resolve(
                    applied, model(true), proposal.membership.inventory(), builder, 0);
                ASSERT_TRUE(resolved.config().moe_routed_expert_plan);
                const auto &plan = *resolved.config().moe_routed_expert_plan;
                ASSERT_EQ(plan.domains.size(), 1u);
                EXPECT_EQ(plan.domains.front().routed_compute_policy, policy);
                for (const auto &domain : plan.dense_domains)
                    EXPECT_EQ(domain.routed_compute_policy, policy);
                EXPECT_EQ(config.routed_expert_compute_policy, policy);
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
    config.routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
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

TEST(AutomaticOrchestrationCandidates, PipelinesKeepCardinalContiguousLayersInBothDirections)
{
    const auto inventory = hosts(2);
    const auto proposals = candidates(request({DeviceType::CPU}, {OrchestrationStrategy::PipelineParallel}), inventory);
    ASSERT_EQ(proposals.size(), 2u);
    std::set<std::pair<int, int>> splits;
    for (const auto &proposal : proposals)
    {
        ASSERT_EQ(proposal.config.pp_stage_definitions.size(), 2u);
        const auto &first = proposal.config.pp_stage_definitions[0];
        const auto &last = proposal.config.pp_stage_definitions[1];
        EXPECT_EQ(first.first_layer, 0);
        EXPECT_EQ(first.last_layer, 1);
        EXPECT_EQ(last.first_layer, 2);
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
        ASSERT_EQ(pp.size(), 12u); // Twelve ordered pairs, one cardinal layer partition.
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
    ASSERT_EQ(proposals.size(), 12u); // Six ROCm subsets, both orders, one cardinal partition.
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
    // Shared visibility adds GPU records after cards() sealed the initial
    // fixture matrices. Publish the same explicit no-peer observations for
    // both observers before any complete rank-local cohort is compiled.
    declareNoPeerTopology(inventory);
    inventory.buildNodeAggregations();

    auto config = request({DeviceType::CUDA, DeviceType::ROCm},
        {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts =
        {{DeviceType::CUDA, 2}, {DeviceType::ROCm, 4}};
    const auto proposals = candidates(config, inventory, true);
    ASSERT_EQ(proposals.size(), 4u); // Both continuation directions/dense policies, with only NUMA-local owners.
    for (const auto &proposal : proposals)
    {
        EXPECT_EQ(proposal.membership.discoveryRanks().size(), 2u);
        const auto &domains = proposal.config.moe_routed_expert_plan->domains;
        ASSERT_EQ(domains.size(), 2u);
        EXPECT_NE(domains[0].owner_rank, domains[1].owner_rank);
        compile(proposal, true);
    }
}

/** @test A node-local CPU cohort retains NUMA separately from the logical CPU ordinal. */
TEST(AutomaticOrchestrationCandidates, MultiTierOverlayTensorShardedCPUUsesCanonicalEndpoints)
{
    auto inventory = hosts(2);
    inventory.ranks[0].cpu.numa_node = 0;
    inventory.ranks[1].cpu.numa_node = 1;
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.ranks[1].local_rank = 1;
    cards(inventory, 0, DeviceType::ROCm, 2);
    auto config = request({DeviceType::CPU, DeviceType::ROCm}, {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts = {{DeviceType::CPU, 2}, {DeviceType::ROCm, 2}};
    config.routed_expert_compute_policy = RoutedExpertComputePolicy::TensorSharded;
    config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;

    const auto proposals = candidates(config, inventory, true);
    ASSERT_FALSE(proposals.empty());
    const auto &q5 = native_quantised_formats::Q5_1;
    const auto format = ExpertWeightFormat::nativeQuantised({q5.codebook_id, q5.is_superblock, true});
    const MoEOverlayLayerWeightManifest source{.layer_idx = 0, .projections = {{
        {ExpertTierWeightProjection::Gate, 640, 512, format},
        {ExpertTierWeightProjection::Up, 640, 512, format},
        {ExpertTierWeightProjection::Down, 512, 640, format}}}};
    for (const auto &proposal : proposals)
    {
        ExecutionPlanBuilder builder;
        const auto resolved = ResolvedRankOrchestration::resolve(
            proposal.config, model(true), proposal.membership.inventory(), builder, 0);
        const auto &plan = *resolved.config().moe_routed_expert_plan;
        const auto targets = configuredMoEOverlayProjectionTargets(plan, source, 32);
        const auto cpu = std::find_if(targets.begin(), targets.end(), [](const auto &target) {
            return target.members.front().device.is_cpu();
        });
        ASSERT_NE(cpu, targets.end());
        ASSERT_EQ(cpu->members.size(), 2u);
        for (int member = 0; member < 2; ++member)
        {
            EXPECT_EQ(cpu->members[member].world_rank, member);
            EXPECT_EQ(cpu->members[member].address,
                      GlobalDeviceAddress::cpu(member, inventory.ranks[member].hostname));
        }
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
    // Shared visibility adds GPU records after cards() sealed the initial
    // fixture matrices. Publish the same explicit no-peer observations for
    // both observers before any complete rank-local cohort is compiled.
    declareNoPeerTopology(inventory);
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

/** @test Default MoE search now uses the installed stage-owned PP authority rather than excluding the family. */
TEST(AutomaticOrchestrationCandidates, DefaultMoESearchIncludesStageOwnedPipelineCandidates)
{
    auto inventory=hosts(1);cards(inventory,0,DeviceType::ROCm,2);
    auto config=request({DeviceType::ROCm},{OrchestrationStrategy::SingleDevice,
        OrchestrationStrategy::TensorParallel,OrchestrationStrategy::PipelineParallel,OrchestrationStrategy::ExpertOverlay});
    const auto proposals=candidates(config,inventory,true);ASSERT_FALSE(proposals.empty());
    std::size_t pipelines=0;
    for(const auto& proposal:proposals) {
        if(proposal.strategy==OrchestrationStrategy::PipelineParallel) {
            ++pipelines;EXPECT_FALSE(proposal.config.moe_routed_expert_plan);
            ASSERT_EQ(proposal.config.pp_stage_definitions.size(),2u);
        } else if(proposal.strategy!=OrchestrationStrategy::SingleDevice) {
            ASSERT_TRUE(proposal.config.moe_routed_expert_plan);
            EXPECT_TRUE(proposal.config.moe_routed_expert_plan->usesExpertOverlayAuthority());
        }
        compile(proposal,true);
    }
    EXPECT_EQ(pipelines,2u);
}

/** @test Positive exact widths retain every disjoint grouping with one cardinal layer partition, including unequal and three-stage plans. */
TEST(AutomaticOrchestrationCandidates, PipelineWidthsAreExactAcrossGeneralPositiveGroups)
{
    for(const auto widths:{std::vector<int>{1,2},std::vector<int>{2,1,1},std::vector<int>{1,1,1,1}}) {
        SCOPED_TRACE(::testing::PrintToString(widths));
        auto inventory=hosts(1);cards(inventory,0,DeviceType::ROCm,4);
        auto config=request({DeviceType::ROCm},{OrchestrationStrategy::PipelineParallel});
        config.automatic_planning.pipeline_stage_widths=widths;
        config.automatic_planning.device_counts=std::vector{AutomaticBackendDeviceCount{DeviceType::ROCm,
            std::accumulate(widths.begin(),widths.end(),0)}};
        const auto proposals=candidates(config,inventory);ASSERT_FALSE(proposals.empty());
        std::set<std::vector<int>> partitions;
        for(const auto& proposal:proposals) {
            ASSERT_EQ(proposal.strategy,OrchestrationStrategy::PipelineParallel);
            ASSERT_EQ(proposal.config.domain_definitions.size(),widths.size());
            std::set<int> physical;
            std::vector<int> boundaries;
            for(std::size_t stage=0;stage<widths.size();++stage) {
                EXPECT_EQ(proposal.config.domain_definitions[stage].devices.size(),std::size_t(widths[stage]));
                for(const auto& device:proposal.config.domain_definitions[stage].devices)
                    EXPECT_TRUE(physical.insert(device.device_ordinal).second);
                const auto& interval=proposal.config.pp_stage_definitions[stage];
                EXPECT_LE(interval.first_layer,interval.last_layer);boundaries.push_back(interval.last_layer);
            }
            EXPECT_EQ(proposal.config.pp_stage_definitions.front().first_layer,0);
            EXPECT_EQ(proposal.config.pp_stage_definitions.back().last_layer,3);
            partitions.insert(boundaries);compile(proposal);
        }
        EXPECT_EQ(partitions.size(),1u);
    }
}

/** @test Exact widths remain general beyond local group sizes; absent switch/P2P facts never prohibit valid native domains. */
TEST(AutomaticOrchestrationCandidates, PipelineWidthsRemainGeneralWithoutSwitchOrDirectPeerFacts)
{
    for(const int width:{1,2,3,4,5,8,9}) {
        SCOPED_TRACE(width);
        auto inventory=hosts(1);cards(inventory,0,DeviceType::ROCm,width+1);
        auto config=request({DeviceType::ROCm},{OrchestrationStrategy::PipelineParallel});
        config.automatic_planning.pipeline_stage_widths=std::vector<int>{width,1};
        config.automatic_planning.device_counts=std::vector{AutomaticBackendDeviceCount{DeviceType::ROCm,width+1}};
        const auto proposals=candidates(config,inventory);ASSERT_FALSE(proposals.empty());
        for(const auto& proposal:proposals) {
            EXPECT_EQ(proposal.config.domain_definitions.front().devices.size(),std::size_t(width));
            EXPECT_EQ(proposal.config.domain_definitions.back().devices.size(),1u);
            EXPECT_EQ(proposal.membership.discoveryRanks(),std::vector<int>{0});
        }
    }
}

/** @test The native CPU TP child retains its original ranks under supported automatic PP. */
TEST(AutomaticOrchestrationCandidates, StaticAutomaticCPUPipelineSelectsOriginalTensorShards)
{
    auto inventory = hosts(3);
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].local_rank = 1;
    cards(inventory, 2, DeviceType::ROCm, 1);

    auto config = request({DeviceType::CPU, DeviceType::ROCm},
        {OrchestrationStrategy::PipelineParallel});
    config.automatic_planning.pipeline_stage_widths = std::vector<int>{2, 1};
    config.automatic_planning.device_counts = {{DeviceType::CPU, 2}, {DeviceType::ROCm, 1}};
    const test::PlanningGGUFFixture native_weights(true, false, GGUFTensorType::Q8_0, 256);
    const test::PlanningGGUFFixture indivisible_weights(true, false, GGUFTensorType::Q4_K, 256);
    const PlanningModelSource native(native_weights.path());
    const PlanningModelSource indivisible(indivisible_weights.path());
    using Compute = RoutedExpertComputePolicy;
    using Maintenance = MoERebalanceRuntimeMode;
    const auto check = [&](const PlanningModelSource &source, Compute requested,
                           Maintenance mode, Compute expected) {
        config.routed_expert_compute_policy = requested;
        config.moe_rebalance.mode = mode;
        std::size_t seen = 0;
        visitAutomaticOrchestrationCandidates(config, source.metadata(), inventory,
            [&](const auto &proposal) {
                ++seen;
                ASSERT_EQ(proposal.membership.discoveryRanks(), (std::vector<int>{0, 1, 2}));
                ASSERT_EQ(proposal.config.domain_definitions.size(), 2u);
                const auto cpu = proposal.config.domain_definitions.front().toExecutionDomainDefinition();
                EXPECT_EQ(cpu.scope, ExecutionDomainScope::NODE_LOCAL);
                EXPECT_EQ(cpu.ranks, (std::vector<int>{0, 1}));
                EXPECT_EQ(cpu.resolveRoutedComputePolicy(1), expected);
                EXPECT_NE(proposal.config.domain_definitions.back().routed_compute_policy, Compute::TensorSharded);
                EXPECT_EQ(proposal.config.moe_rebalance.mode, mode);
                if (expected == Compute::TensorSharded)
                    for (const int rank : {0, 1})
                    {
                        ExecutionPlanBuilder builder;
                        const auto resolved = ResolvedRankOrchestration::resolve(proposal.config,
                            source.metadata(), proposal.membership.inventory(), builder, rank);
                        ASSERT_EQ(resolved.pipelineStages().size(), 1u);
                        const auto &stage = resolved.pipelineStages().front();
                        ASSERT_TRUE(stage.config().moe_routed_expert_plan);
                        const auto &domain = stage.config().moe_routed_expert_plan->domains.front();
                        EXPECT_EQ(domain.routed_compute_policy, Compute::TensorSharded);
                        EXPECT_EQ(domain.world_ranks, (std::vector<int>{0, 1}));
                    }
            });
        EXPECT_GT(seen, 0u);
    };
    check(native, Compute::Automatic, Maintenance::Off, Compute::TensorSharded);
    check(native, Compute::Apportioned, Maintenance::Off, Compute::Apportioned);
    check(native, Compute::Automatic, Maintenance::Dynamic, Compute::Apportioned);
    check(indivisible, Compute::Automatic, Maintenance::Off, Compute::Apportioned);
}

/** @test MoE GPU PP preserves original rank/NUMA, explicit hybrid arithmetic, fixed-head storage and full15 MTP. */
TEST(AutomaticOrchestrationCandidates, RoutedPipelineWidthsRetainGateUpStorageAndDynamicMTP)
{
    auto inventory=hosts(2);cards(inventory,1,DeviceType::ROCm,4);
    auto config=request({DeviceType::ROCm},{OrchestrationStrategy::PipelineParallel});
    config.automatic_planning.pipeline_stage_widths=std::vector<int>{2,2};
    config.automatic_planning.device_counts=std::vector{AutomaticBackendDeviceCount{DeviceType::ROCm,4}};
    config.routed_expert_compute_policy=RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
    config.ngram_table_policy={.storage=ImmutableRowStoragePreference::VRAM,
        .distribution=ImmutableRowTableDistribution::FixedHeadSharded};
    config.mtp.enabled=true;config.mtp.draft_tokens=1;config.mtp.graph_capacity_draft_tokens=15;
    config.mtp.depth_policy.mode=MTPDepthPolicyMode::Dynamic;
    config.mtp.depth_policy.min_depth=1;config.mtp.depth_policy.max_depth=15;
    const auto proposals=candidates(config,inventory,true);ASSERT_FALSE(proposals.empty());
    for(const auto& proposal:proposals) {
        EXPECT_EQ(proposal.membership.discoveryRanks(),std::vector<int>{1});
        EXPECT_FALSE(proposal.config.moe_routed_expert_plan);
        EXPECT_EQ(proposal.config.ngram_table_policy,config.ngram_table_policy);
        EXPECT_EQ(proposal.config.mtp.enabled,config.mtp.enabled);
        EXPECT_EQ(proposal.config.mtp.graph_capacity_draft_tokens,15);
        EXPECT_EQ(proposal.config.mtp.depth_policy.mode,MTPDepthPolicyMode::Dynamic);
        EXPECT_EQ(proposal.config.mtp.depth_policy.min_depth,1);
        EXPECT_EQ(proposal.config.mtp.depth_policy.max_depth,15);
        ExecutionPlanBuilder builder;
        const auto resolved=ResolvedRankOrchestration::resolve(proposal.config,model(true),
            proposal.membership.inventory(),builder,0);
        ASSERT_EQ(resolved.pipelineStages().size(),2u);
        EXPECT_FALSE(resolved.pipelineStages().front().scope().has_lm_head);
        EXPECT_TRUE(resolved.pipelineStages().back().scope().has_lm_head);
        for(const auto& stage:resolved.pipelineStages()) {
            const auto& placement=*stage.config().moe_routed_expert_plan;
            ASSERT_EQ(placement.domains.size(),1u);ASSERT_EQ(placement.routed_tiers.size(),1u);
            EXPECT_EQ(placement.domains.front().routed_compute_policy,RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            EXPECT_EQ(placement.domains.front().owner_rank,0);
            for(const auto& device:placement.domains.front().participants)
                EXPECT_EQ(device.numa_node,inventory.ranks[1].cpu.numa_node);
        }
    }
}

/** @test No fitting ordered groups means no substitute topology; physical output-column constraints remain hard. */
TEST(AutomaticOrchestrationCandidates, PipelineWidthInfeasibilityNeverPublishesAnotherTopology)
{
    auto inventory=hosts(1);cards(inventory,0,DeviceType::ROCm,3);
    auto config=request({DeviceType::ROCm},{OrchestrationStrategy::PipelineParallel});
    config.automatic_planning.pipeline_stage_widths=std::vector<int>{2,2};
    EXPECT_TRUE(candidates(config,inventory,true).empty());
    config.automatic_planning.pipeline_stage_widths=std::vector<int>{1,2};
    config.routed_expert_compute_policy=RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    EXPECT_TRUE(candidates(config,inventory,true).empty());
    config.automatic_planning.pipeline_stage_widths=std::vector<int>{1,1,1,1,1};
    EXPECT_TRUE(candidates(config,inventory).empty());
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

/** @test Automatic candidates retain exact query coverage; independent expert-column constraints remain explicit. */
TEST(AutomaticOrchestrationCandidates, QueryConsumerSourceUsesCanonicalAssignmentsAndPreservesExpertLimits)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(int degree=2;degree<=8;++degree)
    {
        auto inventory=hosts(1);cards(inventory,0,backend,degree);
        auto &observed=inventory.ranks.front();
        observed.cpu.memory_bytes=observed.cpu.free_memory_bytes=observed.cpu_memory_bytes=256ull<<30;
        observed.cpu_execution=test::kSyntheticCPUExecutionGeometry;
        for(auto &gpu:observed.gpus)gpu.memory_bytes=gpu.free_memory_bytes=256ull<<30;
        // A complete authored no-peer matrix is real typed input; absence is not treated as no peer access.
        if(backend==DeviceType::CUDA){observed.p2p_cuda_count=degree;observed.p2p_cuda.assign(degree*degree,false);}
        else {observed.p2p_rocm_count=degree;observed.p2p_rocm.assign(degree*degree,false);}
        auto config=request({backend},{OrchestrationStrategy::TensorParallel});
        config.mtp.enabled=false;config.mtp.graph_capacity_draft_tokens=0;
        config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
        config.routed_expert_compute_policy=RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        config.ngram_table_policy.storage=ImmutableRowStoragePreference::RAM;
        config.automatic_planning.device_counts={{backend,degree}};
        std::size_t proposed=0;
        visitAutomaticOrchestrationCandidates(config,source.metadata(),inventory,[&](const auto &proposal)
        {
            ASSERT_TRUE(proposal.config.moe_routed_expert_plan);
            const auto &placement=*proposal.config.moe_routed_expert_plan;
            if(placement.continuation_domain_spec.effectiveDensePolicy()!=DenseParallelPolicy::TensorParallel)return;
            ++proposed;
            ASSERT_EQ(placement.domains.size(),1u);
            const auto domain=placement.domains.front().toExecutionDomainDefinition();
            ASSERT_EQ(domain.participants.size(),degree);
            EXPECT_EQ(domain.resolveRoutedComputePolicy(placement.routed_tiers.size()),RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            std::vector<DeviceId> members;for(const auto &endpoint:domain.participants)members.push_back(endpoint.toLocalDeviceId());
            const auto declaration=resolveProfileTPDeclaration(source.metadata().memoryProfile(),
                TensorParallelMainPlacement::FullMainRankLocal,config.mtp,
                placement.continuation_domain_spec.effectiveDensePolicy(),members);
            ASSERT_EQ(declaration.coverage(),KVHeadCoveragePolicy::QueryInterval);
            const auto actual=declaration.partition(members,{},source.metadata().memoryProfile().d_ff,
                source.metadata().memoryProfile().vocab_size);
            ExecutionPlanBuilder builder;
            const auto compiled=ResolvedRankOrchestration::resolve(proposal.config,source.metadata(),
                proposal.membership.inventory(),builder,0);
            const auto &plan=compiled.rankPlan();
            // Real overlay admission input, not the ordinary-rank BOM or duplicated capacity arithmetic.
            ASSERT_TRUE(compiled.overlayExecution());
            const auto &installed=*compiled.config().moe_routed_expert_plan;
            const auto policy=resolveMoEOverlayCapacityAdmissionPolicy(installed,compiled.config(),1,48,512);
            const MoEOverlayInferenceGraphFamilyIdentity family{.graph_family_generation=1,.main_layer_count=48,
                .mtp_source_layers={},.max_graph_rows=32,.max_decode_rows=1,.max_request_count=1,.max_mtp_draft_depth=0};
            const std::vector<int> buckets{1,8,32};
            auto input=buildMoEOverlayMemoryPlanInputs({.model=source.metadata().memoryProfile(),.rank_plan=plan,
                .config=compiled.config(),.inventory=proposal.membership.inventory(),.execution=*compiled.overlayExecution(),
                .capacity_policy=policy,.retained_mtp=plan.runtime.mtp,.graph_family=family,
                .prefill={.bucket_rows=buckets,.minimum_sequence_rows=1,.maximum_cached_buckets=16},
                .gpu_weight_load={.maximum_source_bytes=4096}},32);
            // RAM lookup on a GPU declares a CPU service. Supply the same typed
            // fixture evidence as admission tests, without probing the host runtime.
            input.local_capacity.retained_host_services=test::planningHostServiceProjection();
            // Each real RAM-lookup consumer also owns its boundary ordering
            // event. Price those exact authored endpoints through the sealed
            // recorded source; this metadata replay creates no GPU workers.
            input.local_capacity.native_ordering_events=
                test::NativeOrderingEventGroupReplayAccess::projection(members);
            const auto fixed=MoEOverlayLocalCapacityPlanner::plan(input.local_capacity);
            ASSERT_NE(fixed.physical_memory_admission,nullptr);
            const auto &bom=fixed.device_inputs;
            // An overlay may append expert-only physical owners; locate each exact continuation device.
            for(const auto &expected:actual.assignments())
            {
                const auto found=std::find_if(bom.begin(),bom.end(),[&](const auto &entry)
                    {return entry.device==expected.device && entry.tensor_parallel_assignment.has_value();});
                ASSERT_NE(found,bom.end());const auto &local=*found->tensor_parallel_assignment;
                const auto *physical=fixed.physical_memory_admission->plan().find({0,expected.device});
                ASSERT_NE(physical,nullptr);
                EXPECT_GT(physical->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u)
                    << "Original RAM consumer lost its recorded mandatory event contribution";
                EXPECT_EQ(local.head_start,expected.head_start);EXPECT_EQ(local.head_count,expected.head_count);
                EXPECT_EQ(local.kv_head_start,expected.kv_head_start);EXPECT_EQ(local.kv_head_count,expected.kv_head_count);
            }
        });
        const bool expert_columns_fit=MoEProjectionArenaGeometry::hasIntegralOutputPartition(2560,degree);
        EXPECT_EQ(proposed>0,expert_columns_fit)<<"Head coverage is valid1..8; owned expert columns impose a separate source constraint";
    }
}

/** @test Complete visibility must not give a remote rank automatic GPU ownership. */
TEST(AutomaticOrchestrationCandidates, RankLocalGPUOwnershipPreservesDiscoveryAndBothVendors)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto inventory = hosts(2);
        inventory.ranks[0].node_id = inventory.ranks[1].node_id = 23;
        inventory.ranks[0].hostname = inventory.ranks[1].hostname = "shared-host";
        cards(inventory, 1, backend, 2);
        inventory.ranks[0].gpus = inventory.ranks[1].gpus;
        inventory.buildNodeAggregations();
        const auto original = inventory.ranks[0].gpus.size();
        const auto proposals = candidates(request({backend}, {OrchestrationStrategy::SingleDevice}), inventory);
        ASSERT_EQ(proposals.size(), 2u);
        for (const auto &proposal : proposals)
        {
            EXPECT_EQ(proposal.membership.discoveryRanks(), std::vector<int>({1}));
            compile(proposal);
        }
        EXPECT_EQ(inventory.ranks[0].gpus.size(), original);
        EXPECT_EQ(inventory.ranks[1].gpus.size(), original);
    }
}


/** @test Original GPU population never emits zero-head shards, including beyond eight cards. */
TEST(AutomaticOrchestrationCandidates, EveryOriginalTensorParticipantHasPositiveSourceSlicesReplay)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const bool gdn : {false, true})
    {
        auto inventory = hosts(1);
        cards(inventory, 0, backend, 9);
        std::reverse(inventory.ranks.front().gpus.begin(), inventory.ranks.front().gpus.end());
        const auto source = model();
        auto profile = source.memoryProfile();
        profile.n_heads = 8;
        profile.head_dim = profile.d_model / profile.n_heads;
        profile.n_kv_heads = 2;
        profile.gdn_group_count = gdn ? 8 : 0;
        profile.gdn_time_step_rank = gdn ? 8 : 0;
        const auto enumerate = [&](int count) {
            auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
            config.automatic_planning.device_counts = {{backend, count}};
            std::size_t proposals = 0;
            ExecutionPlanBuilder builder;
            visitAutomaticOrchestrationCandidates(config, PlanningModelMetadata(profile, 4), inventory,
                [&](const auto &candidate) {
                    ++proposals;
                    const auto resolved = ResolvedRankOrchestration::resolve(candidate.config,
                        PlanningModelMetadata(profile, 4), candidate.membership.inventory(), builder, 0);
                    EXPECT_TRUE(resolved.rankPlan().validate().empty());
                });
            return proposals;
        };
        for (int replay = 0; replay < 3; ++replay)
        {
            std::size_t infeasible = 1;
            EXPECT_NO_THROW(infeasible = enumerate(9));
            EXPECT_EQ(infeasible, 0u);
            EXPECT_GT(enumerate(8), 0u);
            profile.n_heads = 16;
            profile.head_dim = profile.d_model / profile.n_heads;
            profile.gdn_group_count = gdn ? 16 : 0;
            profile.gdn_time_step_rank = gdn ? 16 : 0;
            if (!gdn) EXPECT_GT(enumerate(9), 0u);
            profile.n_heads = 8;
            profile.head_dim = profile.d_model / profile.n_heads;
            profile.gdn_group_count = gdn ? 8 : 0;
            profile.gdn_time_step_rank = gdn ? 8 : 0;
        }
    }
}

/** @test Cardinal automatic PP is positive, proportional, contiguous and deterministic for arbitrary stage widths. */
TEST(AutomaticOrchestrationCandidates, CardinalPipelineLayerCountsCoverOddAndUnequalStages)
{
    struct Case {int layers;std::vector<int> widths,expected;};
    for(const auto &item:std::vector<Case>{{48,{4,4},{24,24}},{5,{1,1},{3,2}},
        {8,{1,3},{2,6}},{7,{1,3,2},{1,4,2}},{11,{1,2,1},{3,5,3}},
        {3,{100,1,1},{1,1,1}},{1,{9},{1}},{5,{1,1,1,1,1},{1,1,1,1,1}}}) {
        SCOPED_TRACE(::testing::PrintToString(item.widths));
        const auto counts=automaticPipelineLayerCounts(item.layers,item.widths);
        EXPECT_EQ(counts,item.expected);ASSERT_EQ(counts.size(),item.widths.size());
        int next=0;
        for(const int count:counts){EXPECT_GT(count,0);const int first=next;next+=count;EXPECT_EQ(next-first,count);}
        EXPECT_EQ(next,item.layers);
        EXPECT_EQ(automaticPipelineLayerCounts(item.layers,item.widths),counts);
    }
    EXPECT_THROW(automaticPipelineLayerCounts(0,std::vector<int>{1}),std::invalid_argument);
    EXPECT_THROW(automaticPipelineLayerCounts(3,std::vector<int>{}),std::invalid_argument);
    EXPECT_THROW(automaticPipelineLayerCounts(3,std::vector<int>{1,0}),std::invalid_argument);
    EXPECT_THROW(automaticPipelineLayerCounts(3,std::vector<int>{1,-1}),std::invalid_argument);
    EXPECT_THROW(automaticPipelineLayerCounts(1,std::vector<int>{1,1}),std::invalid_argument);
}

/** @test Qwen3.5/3.6 MoE uses its registered graph capability on each native GPU backend. */
TEST(AutomaticOrchestrationCandidates, MultiTierOverlayQwen35MoEUsesTensorShardCapability)
{
    for (const bool with_predictor : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "predictor=" << with_predictor);
        const test::PlanningGGUFFixture file(/*moe=*/true, with_predictor);
        const auto source = readPlanningModelMetadata(file.path());
        ASSERT_EQ(source.memoryProfile().architecture, "qwen35moe");
        ASSERT_FALSE(source.memoryProfile().flash_next);
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            for (const bool mixed : {false, true})
            for (const auto mode : {MoERebalanceRuntimeMode::Off,
                                    MoERebalanceRuntimeMode::Observe,
                                    MoERebalanceRuntimeMode::Dynamic})
            {
                SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend)
                    << " mixed=" << mixed << " mode=" << static_cast<int>(mode));
                const bool sharded = mixed || mode != MoERebalanceRuntimeMode::Observe;
                const auto expected = sharded ? RoutedExpertComputePolicy::TensorSharded
                                              : RoutedExpertComputePolicy::GateUpOwnedDownColumns;
                auto inventory = hosts(mixed ? 2 : 1);
                inventory.ranks[0].cpu.numa_node = 0;
                if (mixed)
                {
                    auto &remote = inventory.ranks[1];
                    remote.cpu.numa_node = 1;
                    remote.node_id = inventory.ranks[0].node_id;
                    remote.hostname = inventory.ranks[0].hostname;
                    remote.local_rank = 1;
                }
                cards(inventory, 0, backend, 2);
                auto config = request(mixed ? std::vector{backend, DeviceType::CPU} : std::vector{backend},
                    {mixed ? OrchestrationStrategy::ExpertOverlay : OrchestrationStrategy::TensorParallel});
                config.automatic_planning.device_counts = {{backend, 2}};
                config.moe_rebalance.mode = mode;
                if (mixed)
                    config.automatic_planning.device_counts->emplace_back(DeviceType::CPU, 2);

                int candidates = 0;
                visitAutomaticOrchestrationCandidates(config, source, inventory, [&](const auto &candidate) {
                    ++candidates;
                    const auto &authored = *candidate.config.moe_routed_expert_plan;
                    ASSERT_EQ(authored.domains.size(), mixed ? 2u : 1u);
                    if (sharded)
                        EXPECT_EQ(authored.continuation_domain_spec.effectiveDensePolicy(),
                                  DenseParallelPolicy::TensorParallel);
                    EXPECT_EQ(candidate.config.moe_rebalance.mode, mode);
                    ExecutionPlanBuilder builder;
                    const auto resolved = ResolvedRankOrchestration::resolve(
                        candidate.config, source, candidate.membership.inventory(), builder, 0);
                    EXPECT_EQ(resolved.rankPlan().runtime.mtp.enabled, with_predictor);
                    if (with_predictor)
                    {
                        EXPECT_EQ(resolved.rankPlan().runtime.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
                        auto ordinary = candidate.config;
                        ordinary.mtp_activation_policy = MTPActivationPolicy::Disabled;
                        const auto without_mtp = ResolvedRankOrchestration::resolve(
                            ordinary, source, candidate.membership.inventory(), builder, 0);
                        EXPECT_FALSE(without_mtp.rankPlan().runtime.mtp.enabled);
                        EXPECT_EQ(without_mtp.config().moe_routed_expert_plan->domains.front().routed_compute_policy,
                                  expected);
                    }
                    const auto &bound = *resolved.config().moe_routed_expert_plan;
                    for (const auto &domain : bound.domains)
                    {
                        EXPECT_EQ(domain.routed_compute_policy, expected);
                        EXPECT_EQ(domain.participants.size(), 2u);
                    }
                });
                EXPECT_EQ(candidates, sharded ? 1 : 2);
            }
    }
}

/** @test Non-QFN routed down-K keeps complete source-native blocks at every layer. */
TEST(AutomaticOrchestrationCandidates, MultiTierOverlayChecksOriginalDownNativeBlocks)
{
    const test::PlanningGGUFFixture file(/*moe=*/true, /*mtp=*/false);
    const auto source = readPlanningModelMetadata(file.path());
    auto profile = source.memoryProfile();
    std::vector<std::size_t> down_indices;
    for (std::size_t i = 0; i < profile.tensors.size(); ++i)
        if (source.classifiedTensor(i).role() == WeightRole::MoEExpertDown)
            down_indices.push_back(i);
    ASSERT_GE(down_indices.size(), 2u);
    for (const auto i : down_indices)
    {
        profile.tensors[i].K = 512;
        profile.tensors[i].quant_type = "Q8_0";
    }
    const PlanningModelMetadata q8(profile, source.mainLayerCount());
    EXPECT_TRUE(supportsOriginalRoutedIntermediateShardDegree(q8, 4, true));

    // One layer with 256-value source blocks bounds the whole retained graph family.
    profile.tensors[down_indices.back()].quant_type = "IQ4_XS";
    const PlanningModelMetadata mixed(profile, source.mainLayerCount());
    for (const bool gpu : {false, true})
    {
        EXPECT_TRUE(supportsOriginalRoutedIntermediateShardDegree(mixed, 2, gpu));
        EXPECT_FALSE(supportsOriginalRoutedIntermediateShardDegree(mixed, 4, gpu));
    }
    auto inventory = hosts(1);
    cards(inventory, 0, DeviceType::ROCm, 4);
    auto config = request({DeviceType::ROCm}, {OrchestrationStrategy::TensorParallel});
    config.automatic_planning.device_counts = {{DeviceType::ROCm, 4}};
    config.routed_expert_compute_policy = RoutedExpertComputePolicy::Automatic;
    int emitted = 0;
    EXPECT_THROW(visitAutomaticOrchestrationCandidates(config, mixed, inventory,
        [&](const auto &) { ++emitted; }), std::invalid_argument);
    EXPECT_EQ(emitted, 0);
}

/** @test Original shared Down blocks bind the same positive intervals for every retained layer. */
TEST(AutomaticOrchestrationCandidates, OriginalSharedExpertTPKeepsNativeBlocksAcrossMTP)
{
    using Format = GGUFTensorType;
    const test::PlanningGGUFFixture file(true, true, Format::Q8_0, 512, std::nullopt,
        8, test::PlanningGGUFFixture::Geometry::FullAttentionGraph, 2, 8,
        Format::Q8_0, std::nullopt, std::nullopt, std::nullopt, Format::Q6_K);
    const auto source = readPlanningModelMetadata(file.path());
    auto profile = source.memoryProfile();
    profile.d_ff = 1; // A generic metadata heuristic cannot size the shared matrix.
    ASSERT_EQ(profile.mtp_layer_count, 1);
    ASSERT_EQ(profile.totalLayerCount(), 3);
    OrchestrationConfig config_with_mtp;
    resolveMTPStartupPolicy(config_with_mtp, profile);
    const auto geometry = originalSharedExpertTPGeometry(profile, config_with_mtp.mtp);
    ASSERT_TRUE(geometry);
    EXPECT_EQ(geometry->columns, 512);
    EXPECT_EQ(geometry->group, 256);
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const std::vector devices{DeviceId(backend, 7), DeviceId(backend, 3)};
        const auto original = TensorParallelConfig::proportionalSplit(devices, {0.625f, 0.375f},
            profile.n_heads, profile.n_kv_heads, 2048, profile.vocab_size);
        const auto plan = bindOriginalSharedExpertTP(profile, config_with_mtp.mtp, original);
        for (int participant = 0; participant < 2; ++participant)
        {
            const auto &interval = plan.forRank(participant);
            EXPECT_EQ(interval.device, devices[participant]);
            EXPECT_EQ(interval.head_count, original.forRank(participant).head_count);
            EXPECT_EQ(interval.d_ff_start, participant * 256);
            EXPECT_EQ(interval.d_ff_count, 256);
            for (int layer = 0; layer < profile.totalLayerCount(); ++layer)
                EXPECT_NO_THROW(MoEIntermediateShardArenaGeometry::resolve(profile, 2, participant,
                    interval, layer, layer));
            const WeightShardGeometryResolver priced(profile, devices[participant], participant, 2, interval);
            int priced_shared_roles = 0;
            for (const auto &tensor : profile.tensors)
            {
                const auto role = inferWeightRole(tensor.name);
                const bool down = role == WeightRole::SharedExpertDown;
                if (!down && role != WeightRole::SharedExpertGate && role != WeightRole::SharedExpertUp)
                    continue;
                const auto local = priced.resolve(tensor);
                ASSERT_TRUE(local.matrix());
                EXPECT_EQ(down ? local.matrix()->columns : local.matrix()->rows, 256u);
                EXPECT_EQ(down ? local.matrix()->rows : local.matrix()->columns, profile.d_model);
                ++priced_shared_roles;
            }
            EXPECT_EQ(priced_shared_roles, profile.totalLayerCount() * 3);
        }

        auto inventory = hosts(1);
        cards(inventory, 0, backend, 4);
        auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
        config.automatic_planning.device_counts = {{backend, 4}};
        EXPECT_THROW(visitAutomaticOrchestrationCandidates(config, source, inventory,
            [&](auto) { ADD_FAILURE() << "Published an indivisible shared Down partition"; }), std::invalid_argument);
    }
}

/** @test A real ModelContext loader adapter still exposes the original shared routes to TP preparation. */
TEST(AutomaticOrchestrationCandidates, OriginalSharedExpertTPBindsThroughProductionLoaderAdapter)
{
    const test::Qwen4ExpPlanningGGUFFixture file;
    ModelContextConfig configuration;
    configuration.use_mmap = false;
    auto original = ModelContext::create(file.path(), configuration);
    ASSERT_NE(original, nullptr);
    IModelContext &interface = *original;
    ASSERT_NE(interface.loader(), nullptr);
    EXPECT_EQ(std::dynamic_pointer_cast<ModelLoader>(interface.loader()), nullptr);
    const auto profile = ModelMemoryProfile::fromOriginalModelContext(interface);
    EXPECT_EQ(profile.architecture, "qwen4exp");
    const auto &main = original->concreteLoader();
    const auto parsed = ModelMemoryProfile::fromLoader(main);
    OrchestrationConfig off;
    off.mtp_activation_policy = MTPActivationPolicy::Disabled;
    resolveMTPStartupPolicy(off, profile);
    const std::vector devices{DeviceId::rocm(1), DeviceId::rocm(0)};
    const auto original_tp = TensorParallelConfig::proportionalSplit(
        devices, {0.5f, 0.5f}, profile.n_heads, profile.n_kv_heads, 640, profile.vocab_size);
    const auto shared = bindOriginalSharedExpertTP(profile, off.mtp, original_tp);
    EXPECT_EQ(shared.assignments(), bindOriginalSharedExpertTP(parsed, off.mtp, original_tp).assignments());
    EXPECT_EQ(shared.forRank(0).d_ff_count, 320);
    EXPECT_EQ(shared.forRank(1).dFFEnd(), 640);
}

/** @test Paired shared source grain follows retained materialization rather than discovery. */
TEST(AutomaticOrchestrationCandidates, OriginalSharedExpertTPUsesOnlyMaterializedPairedPredictor)
{
    using Format = GGUFTensorType;
    test::Qwen4ExpSidecarFixture fixture;
    constexpr uint32_t width = 640;
    for (int layer = 0; layer < 48; ++layer)
    {
        auto &down = fixture.main.tensors.emplace_back(GGUFTensorInfo{
            .name = "blk." + std::to_string(layer) + ".ffn_down_shexp.weight",
            .dimensions = {2560, width}, .type = Format::Q8_0});
        test::Qwen4ExpSidecarFixture::sealBytes(down);
    }
    const auto compatible = ModelMemoryProfile::fromGGUF(fixture.main, fixture.predictor,
        SharedPredictorMaterialization::Active);
    auto &predictor_down = *std::find_if(fixture.predictor.tensors.begin(), fixture.predictor.tensors.end(),
        [](const auto &tensor) { return tensor.name.ends_with("ffn_down_shexp.weight"); });
    predictor_down.type = Format::Q6_K;
    test::Qwen4ExpSidecarFixture::sealBytes(predictor_down);

    const auto inactive = ModelMemoryProfile::fromGGUF(fixture.main, fixture.predictor,
        SharedPredictorMaterialization::Inactive);
    const auto active = ModelMemoryProfile::fromGGUF(fixture.main, fixture.predictor,
        SharedPredictorMaterialization::Active);
    ASSERT_EQ(inactive.mtp_layer_count, 1);
    ASSERT_EQ(inactive.totalLayerCount(), active.totalLayerCount());
    OrchestrationConfig off, on;
    off.mtp_activation_policy = MTPActivationPolicy::Disabled;
    resolveMTPStartupPolicy(off, inactive);
    resolveMTPStartupPolicy(on, active);
    ASSERT_FALSE(retainsMTPGraphCapacity(off.mtp));
    ASSERT_TRUE(retainsMTPGraphCapacity(on.mtp));
    const auto off_geometry = originalSharedExpertTPGeometry(inactive, off.mtp);
    const auto active_source_off = originalSharedExpertTPGeometry(active, off.mtp);
    const auto on_geometry = originalSharedExpertTPGeometry(active, on.mtp);
    ASSERT_TRUE(off_geometry);
    ASSERT_TRUE(active_source_off);
    ASSERT_TRUE(on_geometry);
    EXPECT_EQ(off_geometry->group, 32);
    EXPECT_EQ(active_source_off->group, off_geometry->group);
    EXPECT_TRUE(off_geometry->supportsDegree(4));
    EXPECT_EQ(on_geometry->group, 256);
    EXPECT_FALSE(on_geometry->supportsDegree(4));
    EXPECT_FALSE(originalSharedExpertTPGeometry(inactive, on.mtp));

    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const std::vector devices{DeviceId(backend, 7), DeviceId(backend, 3)};
        const auto original = TensorParallelConfig::proportionalSplit(devices, {0.5f, 0.5f},
            active.n_heads, active.n_kv_heads, width, active.vocab_size);
        const auto serial = bindOriginalSharedExpertTP(inactive, off.mtp, original);
        EXPECT_EQ(serial.assignments(), bindOriginalSharedExpertTP(active, off.mtp, original).assignments());
        EXPECT_EQ(serial.forRank(0).d_ff_count, 320);
        const auto retained = bindOriginalSharedExpertTP(compatible, on.mtp, original);
        EXPECT_EQ(retained.forRank(0).device, devices.front());
        EXPECT_EQ(retained.forRank(1).device, devices.back());
        EXPECT_EQ(retained.forRank(1).dFFEnd(), width);
        EXPECT_EQ(retained.assignments(), serial.assignments());
        EXPECT_THROW((void)bindOriginalSharedExpertTP(active, on.mtp, original), std::invalid_argument);
    }
}

/** @test The same serial MoE roots cover the actually materialized shared predictor. */
TEST(AutomaticOrchestrationCandidates, IntermediateShardArenaUsesOwnedPredictorSharedWidth)
{
    using Fixture = test::PlanningGGUFFixture;
    const Fixture file(true, true, GGUFTensorType::Q8_0, 640, std::nullopt, 8,
        Fixture::Geometry::FullAttentionGraph, 2, 8, GGUFTensorType::Q8_0,
        std::nullopt, Fixture::MoEDimensions{.hidden_columns = 2560, .shared_columns = 640, .top_k = 2});
    const auto metadata = readPlanningModelMetadata(file.path());
    const auto &source = metadata.memoryProfile();
    ASSERT_EQ(source.mainLayerCount(), 2);
    ASSERT_EQ(source.totalLayerCount(), 3);
    for (const auto vendor : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto policy = TensorParallelConfig::proportionalSplit(
            {DeviceId(vendor, 3), DeviceId(vendor, 0)}, {0.6f, 0.4f},
            source.n_heads, source.n_kv_heads, 640, source.vocab_size);
        const auto &shared = policy.forRank(0);
        ASSERT_EQ(shared.d_ff_count, 384);
        const auto exactWidth = [&](int first, int last, MTPShiftedKVHeadLayout layout) {
            const auto geometry = MoEIntermediateShardArenaGeometry::resolve(source, 2, 0, shared,
                first, last, GPUExpertIntermediateTiling::Coarse128, layout);
            GraphResolverConfig resolver;
            geometry.bindResolver(resolver);
            EXPECT_EQ(resolver.custom_formulas.at("moe_ffn_intermediate_max"), geometry.scratchIntermediate());
            EXPECT_EQ(geometry.sharedIntermediate(), 384);
            EXPECT_LE(geometry.routedIntermediate(), 384);
            EXPECT_EQ(geometry.routeSlots(), source.expert_used_count);
            return geometry.scratchIntermediate();
        };
        const auto replicated = MTPShiftedKVHeadLayout::FullModelPerParticipant;
        const auto sharded = MTPShiftedKVHeadLayout::PrimaryTensorParallelShard;
        EXPECT_EQ(exactWidth(0, 2, replicated), 640);
        EXPECT_EQ(exactWidth(2, 2, replicated), 640); // PP child owning the retained predictor.
        EXPECT_EQ(exactWidth(0, 1, replicated), 384); // Another PP child cannot price its roots.
        EXPECT_EQ(exactWidth(0, 2, sharded), 384); // The sharded policy retains local arithmetic.
    }
}

/** @test No layout flags are needed to select the original QFN shard cohorts. */
TEST(AutomaticOrchestrationCandidates, QFNAutomaticTensorShardsHonorTopologyAndOptOuts)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto &qfn = source.metadata();
    OrchestrationConfig without_predictor;
    resolveMTPStartupPolicy(without_predictor, qfn.memoryProfile());
    EXPECT_FALSE(without_predictor.mtp.enabled);
    OrchestrationConfig with_predictor;
    const auto predictor = model(true);
    resolveMTPStartupPolicy(with_predictor, predictor.memoryProfile());
    EXPECT_TRUE(with_predictor.mtp.enabled);
    EXPECT_EQ(with_predictor.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
    OrchestrationConfig disabled;
    disabled.mtp_activation_policy = MTPActivationPolicy::Disabled;
    resolveMTPStartupPolicy(disabled, predictor.memoryProfile());
    EXPECT_FALSE(disabled.mtp.enabled);

    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        for (const int degree : {2, 3, 4, 8})
        {
            auto inventory = hosts(1);
            cards(inventory, 0, backend, degree);
            auto config = request({backend}, {OrchestrationStrategy::TensorParallel});
            config.automatic_planning.device_counts = {{backend, degree}};
            int selected = 0;
            visitAutomaticOrchestrationCandidates(config, qfn, inventory, [&](const auto &candidate) {
                ++selected;
                const auto &plan = *candidate.config.moe_routed_expert_plan;
                ASSERT_EQ(plan.domains.size(), 1u);
                EXPECT_EQ(plan.domains.front().participants.size(), degree);
                EXPECT_EQ(plan.domains.front().routed_compute_policy, RoutedExpertComputePolicy::TensorSharded);
                EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
                EXPECT_EQ(candidate.config.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
                EXPECT_EQ(plan.residency_policy, RoutedExpertResidencyPolicy::RoutedTierRebalanced);
            });
            EXPECT_EQ(selected, 1) << degree;
        }
    }

    auto inventory = hosts(2);
    inventory.ranks[0].cpu.numa_node = 0;
    inventory.ranks[1].cpu.numa_node = 1;
    inventory.ranks[1].node_id = inventory.ranks[0].node_id;
    inventory.ranks[1].hostname = inventory.ranks[0].hostname;
    inventory.ranks[1].local_rank = 1;
    cards(inventory, 0, DeviceType::ROCm, 2);
    auto config = request({DeviceType::CPU, DeviceType::ROCm}, {OrchestrationStrategy::ExpertOverlay});
    config.automatic_planning.device_counts = {{DeviceType::CPU, 2}, {DeviceType::ROCm, 2}};
    for (const auto compute : {RoutedExpertComputePolicy::Automatic, RoutedExpertComputePolicy::Apportioned})
    {
        config.routed_expert_compute_policy = compute;
        if (compute == RoutedExpertComputePolicy::Apportioned)
        {
            config.mtp_activation_policy = MTPActivationPolicy::Disabled;
            config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        }
        int selected = 0;
        visitAutomaticOrchestrationCandidates(config, qfn, inventory, [&](const auto &candidate) {
            ++selected;
            const auto &plan = *candidate.config.moe_routed_expert_plan;
            ASSERT_EQ(plan.domains.size(), 2u);
            const bool automatic = compute == RoutedExpertComputePolicy::Automatic;
            EXPECT_EQ(plan.residency_policy, automatic ? RoutedExpertResidencyPolicy::RoutedTierRebalanced
                                                       : RoutedExpertResidencyPolicy::StaticById);
            EXPECT_EQ(candidate.config.mtp_activation_policy, config.mtp_activation_policy);
            for (const auto &domain : plan.domains)
            {
                EXPECT_EQ(domain.participants.size(), 2u);
                EXPECT_EQ(domain.routed_compute_policy, automatic ? RoutedExpertComputePolicy::TensorSharded : compute);
            }
        });
        EXPECT_GT(selected, 0);
    }
}
