/**
 * @file Test__MoEProjectionPreparation.cpp
 * @brief Device-free proofs of projection-specific expert preparation.
 *
 * Gate/up ownership still comes from the existing overlay owner map. Down
 * residency covers every expert but only the participant's fixed output slice.
 * These tests deliberately separate physical device ordinals, global domain
 * coordinates and MPI ranks so filtering cannot accidentally reshard weights.
 */
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEExpertOverlayPreparationPlan.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "mocks/MockModelLoader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <numeric>
#include <set>

namespace llaminar2::test
{
namespace
{
    using Role = ExpertGemmRegistry::WeightRole;
    constexpr MoEExpertProjectionOwnership::Geometry geometry{24, 1536, 256};

    /** @brief Directory-only oracle; even large geometry allocates no weight payload. */
    class SourceDirectory final : public MockModelLoader
    {
    public:
        /** @brief Publish complete GGUF axes independently of the preparation compiler. */
        explicit SourceDirectory(MoEExpertProjectionOwnership::Geometry shape)
        {
            for (const int layer : {0, 3})
            {
                const auto prefix = "blk." + std::to_string(layer) + ".";
                for (const auto *role : {"ffn_gate_exps.weight", "ffn_up_exps.weight"})
                    shapes[prefix + role] = {static_cast<size_t>(shape.model_columns),
                        static_cast<size_t>(shape.intermediate_columns), static_cast<size_t>(shape.experts)};
                shapes[prefix + "ffn_down_exps.weight"] = {static_cast<size_t>(shape.intermediate_columns),
                    static_cast<size_t>(shape.model_columns), static_cast<size_t>(shape.experts)};
            }
        }
        /** @return Only fixture directory metadata; no tensor construction or device work. */
        std::optional<std::vector<size_t>> getTensorShape(const std::string &name) const override
        {
            const auto found = shapes.find(name);
            return found == shapes.end() ? std::nullopt : std::optional(found->second);
        }
        std::map<std::string, std::vector<size_t>> shapes;
    };

    /** @brief Construct metadata only; no backend or driver may be initialized. */
    std::shared_ptr<MoERoutedExpertPlacementPlan> planFor(
        bool rocm, int participants, RoutedExpertOwnerOrder order, int experts = geometry.experts)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::SingleDomain;
        plan->continuation_domain = "compute";
        plan->shared_expert_domain = "compute";
        plan->residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan->owner_order = order;
        RoutedExpertDomain domain;
        domain.name = "compute";
        domain.scope = participants == 1 ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::RANK_LOCAL;
        domain.owner_rank = 0;
        domain.backend = rocm ? CollectiveBackendType::RCCL : CollectiveBackendType::NCCL;
        domain.routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        for (int index = 0; index < participants; ++index)
        {
            // Physical IDs run backwards and do not start at zero. A slice's
            // coordinate must come from domain order, not the ordinal itself.
            const int ordinal = participants + 2 - index;
            domain.participants.push_back(rocm ? GlobalDeviceAddress::rocm(ordinal)
                                               : GlobalDeviceAddress::cuda(ordinal));
        }
        plan->domains.push_back(std::move(domain));
        plan->routed_tiers.push_back(RoutedExpertTier{.name = "capacity", .domain = "compute", .priority = 7});
        for (const int layer : {0, 3})
            plan->placements.push_back({.layer = layer, .routed_expert_tier = std::vector<int>(experts, 0)});
        return plan;
    }

    /**
     * @brief Change only the frozen physical mode for an independent A/B control.
     * @param runtime Baseline topology, exact rank coordinates and owner ordering.
     * @return A distinct whole-expert plan; no live graph or residency is mutated.
     */
    std::shared_ptr<MoEExpertOverlayRuntimePlan> wholeRuntime(const MoEExpertOverlayRuntimePlan &runtime)
    {
        auto source = std::make_shared<MoERoutedExpertPlacementPlan>(runtime.sourcePlan());
        for (auto &domain : source->domains)
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return resolveMoEExpertOverlayRuntimePlan(source,
            {.current_world_rank = runtime.currentWorldRank(), .validate_mvp_root_reachability = false});
    }

    /** @brief Verify that filtering preserves every source/slice identity byte-for-byte. */
    void expectPreserved(const MoEExpertOverlayPreparationPlan &before,
                         const MoEExpertOverlayPreparationPlan &after)
    {
        for (const auto &request : after.requests())
        {
            const auto *original = before.requestForParticipant(
                request.domain_name, request.device, request.participant_world_rank,
                request.participant_index, request.layer, request.expert_id, request.role);
            ASSERT_NE(original, nullptr);
            EXPECT_EQ(request.projection_ownership, original->projection_ownership);
        }
    }
}

TEST(MoEProjectionPreparation, PreservesCanonicalOwnersAndPreparesEveryDownSlice)
{
    const SourceDirectory directory(geometry);
    for (const bool rocm : {false, true})
        for (const int participants : {2, 3, 4, 8})
            for (const auto order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
            {
                SCOPED_TRACE(::testing::Message() << "rocm=" << rocm << " participants=" << participants
                                                 << " order=" << static_cast<int>(order));
                const auto runtime = resolveMoEExpertOverlayRuntimePlan(planFor(rocm, participants, order));
                const auto whole = MoEExpertOverlayPreparationPlan::build(*wholeRuntime(*runtime), directory, 123);
                const auto partitioned = MoEExpertOverlayPreparationPlan::build(*runtime, directory);
                const auto owner_map = MoEExpertOwnerMap::build(runtime->sourcePlan());
                for (const auto &endpoint : owner_map.participants())
                {
                    for (const int layer : {0, 3})
                    {
                        const auto layout = partitioned.requireProjectionOwnershipForParticipant(endpoint, layer);
                        EXPECT_EQ(layout.geometry(), geometry);
                        EXPECT_EQ(layout.participant(), endpoint.domain_participant_index);
                        EXPECT_EQ(layout.participants(), participants);
                        EXPECT_THROW(whole.requireProjectionOwnershipForParticipant(endpoint, layer), std::invalid_argument);
                    }
                    EXPECT_THROW(partitioned.requireProjectionOwnershipForParticipant(endpoint, 2), std::invalid_argument);
                    auto wrong = endpoint;
                    wrong.world_rank = 99;
                    wrong.world_rank_known = true;
                    EXPECT_THROW(partitioned.requireProjectionOwnershipForParticipant(wrong, 0), std::invalid_argument);
                }
                EXPECT_EQ(partitioned.requests().size(), 2u * geometry.experts * (2 + participants));
                std::vector<int> all_experts(geometry.experts);
                std::iota(all_experts.begin(), all_experts.end(), 0);
                for (const auto &participant : runtime->domainForTier(0).participants)
                {
                    const auto device = participant.local_device;
                    const auto scoped = partitioned.filteredForDevice(device);
                    expectPreserved(partitioned, scoped);
                    size_t owners = 0;
                    for (const int layer : {0, 3})
                    {
                        const auto expected = whole.expertsForDeviceLayerRole(device, layer, Role::GATE);
                        owners += expected.size();
                        EXPECT_EQ(scoped.expertsForDeviceLayerRole(device, layer, Role::GATE), expected);
                        EXPECT_EQ(scoped.expertsForDeviceLayerRole(device, layer, Role::UP), expected);
                        EXPECT_EQ(scoped.expertsForDeviceLayerRole(device, layer, Role::DOWN), all_experts);
                    }
                    ASSERT_EQ(scoped.diagnostics().domains.size(), 1u);
                    const auto &stats = scoped.diagnostics().domains.front();
                    EXPECT_EQ(stats.assigned_routed_experts, owners);
                    EXPECT_EQ(stats.planned_engine_count, 2 * owners + 2 * geometry.experts);
                    EXPECT_EQ(stats.estimated_routed_bytes, 0u); // PMA prices the role-aware BOM.
                    for (const auto &request : scoped.requests())
                    {
                        ASSERT_TRUE(request.projection_ownership.has_value());
                        const auto &ownership = *request.projection_ownership;
                        EXPECT_EQ(ownership.geometry(), geometry);
                        EXPECT_EQ(ownership.participant(), participant.participant_index);
                        EXPECT_EQ(ownership.participants(), participants);
                        const auto down = ownership.projection(WeightRole::MoEExpertDown);
                        EXPECT_EQ(down.rows, geometry.model_columns / participants);
                        EXPECT_EQ(down.first_row, participant.participant_index * down.rows);
                        EXPECT_EQ(down.source_columns, geometry.intermediate_columns);
                    }
                }
                for (const auto &request : whole.requests())
                    EXPECT_FALSE(request.projection_ownership.has_value());
            }
}

/** @test Neither metadata omission nor preparation dispatch may silently change the A/B mode. */
TEST(MoEProjectionPreparation, FrozenComputeModeIsTheSolePreparationSelector)
{
    const SourceDirectory directory(geometry);
    const auto projected = resolveMoEExpertOverlayRuntimePlan(planFor(true, 2, RoutedExpertOwnerOrder::Random));
    const auto whole = wholeRuntime(*projected);
    EXPECT_NE(projected->diagnostics(), whole->diagnostics());
    EXPECT_STREQ(routedExpertComputePolicyToString(projected->domainForTier(0).routed_compute_policy),
        "gate-up-owned-down-columns");
    EXPECT_STREQ(routedExpertComputePolicyToString(whole->domainForTier(0).routed_compute_policy), "apportioned");
    EXPECT_THROW(MoEExpertOverlayPreparationPlan::build(*projected), std::invalid_argument);
    EXPECT_THROW(MoEExpertOverlayPreparationPlan::build(*projected, 1024), std::invalid_argument);
    const auto a = MoEExpertOverlayPreparationPlan::build(*whole, directory);
    const auto b = MoEExpertOverlayPreparationPlan::build(*projected, directory);
    ASSERT_FALSE(a.empty());
    ASSERT_FALSE(b.empty());
    for (const auto &request : a.requests()) EXPECT_FALSE(request.projection_ownership);
    for (const auto &request : b.requests()) ASSERT_TRUE(request.projection_ownership);
    // Even a bogus legacy whole-expert diagnostic price must not become a
    // discount applied to the new mode's physical memory BOM.
    const auto priced = MoEExpertOverlayPreparationPlan::build(*projected, directory, 1024);
    for (const auto &stats : priced.diagnostics().domains) EXPECT_EQ(stats.estimated_routed_bytes, 0u);
}

/** The native multi-GPU domain may belong to any MPI rank, not necessarily rank zero. */
TEST(MoEProjectionPreparation, NonzeroOwnerRankRetainsDomainPartitionCoordinates)
{
    const SourceDirectory directory(geometry);
    for (const bool rocm : {false, true})
        for (const auto order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
        {
            auto plan = planFor(rocm, 3, order);
            auto &domain = plan->domains.front();
            domain.owner_rank = 2;
            plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
            for (auto &address : domain.participants)
                address.hostname = "bound-node";
            const auto execution = resolveMoEExpertOverlayExecutionPlan(plan,
                MoEExpertOverlayExecutionPlanResolverOptions{.current_world_rank = 0, .world_size = 3});
            for (int rank = 0; rank < 3; ++rank)
            {
                SCOPED_TRACE(::testing::Message() << "rocm=" << rocm << " rank=" << rank);
                const auto runtime = resolveMoEExpertOverlayRuntimePlan(plan,
                    MoEExpertOverlayRuntimeResolverOptions{.current_world_rank = rank,
                        .validate_mvp_root_reachability = false});
                const auto whole = MoEExpertOverlayPreparationPlan::build(*wholeRuntime(*runtime), directory);
                const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime, directory);
                if (rank != domain.owner_rank)
                {
                    EXPECT_TRUE(prep.empty());
                    continue;
                }
                const auto *rank_plan = execution.rankPlanFor(rank);
                ASSERT_NE(rank_plan, nullptr);
                const auto filtered = prep.filteredForRank(*rank_plan);
                EXPECT_EQ(filtered.requests().size(), prep.requests().size());
                expectPreserved(prep, filtered);
                for (const auto &request : filtered.requests())
                {
                    EXPECT_EQ(request.participant_world_rank, rank);
                    const auto &participant = runtime->domainForTier(0).participants.at(request.participant_index);
                    EXPECT_EQ(request.device, participant.local_device);
                    ASSERT_TRUE(request.projection_ownership.has_value());
                    EXPECT_EQ(request.projection_ownership->participant(), request.participant_index);
                    EXPECT_EQ(request.projection_ownership->participants(), 3);
                    if (request.role != Role::DOWN)
                        EXPECT_TRUE(whole.shouldPrepare(request.device, request.layer, request.expert_id, request.role));
                }
                ASSERT_EQ(filtered.diagnostics().domains.size(), 3u);
                for (const auto &diagnostics : filtered.diagnostics().domains)
                {
                    EXPECT_EQ(diagnostics.assigned_routed_experts, 16u);
                    EXPECT_EQ(diagnostics.planned_engine_count, 80u);
                }
                const auto sources = filtered.sourceWeightPlan(directory, ModelContextId{51});
                ASSERT_EQ(sources.size(), 18u);
                for (const auto &source : sources.requirements())
                {
                    EXPECT_EQ(source.overlay_participant_world_rank, rank);
                    EXPECT_EQ(source.target_device,
                        runtime->domainForTier(0).participants.at(source.overlay_participant_index).local_device);
                    EXPECT_EQ(source.slice.expert_count, source.role == WeightRole::MoEExpertDown ? 24u : 8u);
                }
            }
        }
}

/** Geometry support alone must never advertise an unimplemented production topology. */
TEST(MoEProjectionPreparation, RejectsSingleDeviceAndCrossRankExecutionBeforePreparation)
{
    for (const bool rocm : {false, true})
    {
        EXPECT_THROW(resolveMoEExpertOverlayRuntimePlan(planFor(rocm, 1, RoutedExpertOwnerOrder::Ordinal)),
                     std::invalid_argument);
        for (const auto scope : {ExecutionDomainScope::NODE_LOCAL, ExecutionDomainScope::GLOBAL})
        {
            auto plan = planFor(rocm, 3, RoutedExpertOwnerOrder::Random);
            auto &domain = plan->domains.front();
            domain.scope = scope;
            domain.owner_rank = 2;
            domain.world_ranks = {2, 0, 1};
            plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
            EXPECT_THROW(resolveMoEExpertOverlayRuntimePlan(plan), std::invalid_argument);
            EXPECT_THROW(resolveMoEExpertOverlayExecutionPlan(plan,
                (MoEExpertOverlayExecutionPlanResolverOptions{.current_world_rank = 0, .world_size = 3})),
                std::invalid_argument);
        }
    }
}

TEST(MoEProjectionPreparation, ZeroGateUpOwnersStillPrepareDown)
{
    const SourceDirectory directory({1, 512, 256});
    for (const bool rocm : {false, true})
    {
        const auto runtime = resolveMoEExpertOverlayRuntimePlan(planFor(rocm, 4, RoutedExpertOwnerOrder::Ordinal, 1));
        const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime, directory);
        size_t empty_owners = 0;
        for (const auto &participant : runtime->domainForTier(0).participants)
        {
            const auto filtered = prep.filteredForDevice(participant.local_device);
            ASSERT_EQ(filtered.diagnostics().domains.size(), 1u);
            const auto &stats = filtered.diagnostics().domains.front();
            if (stats.assigned_routed_experts == 0)
            {
                ++empty_owners;
                EXPECT_EQ(filtered.requests().size(), 2u);
                EXPECT_EQ(stats.planned_engine_count, 2u);
                for (const auto &request : filtered.requests())
                    EXPECT_EQ(request.role, Role::DOWN);
                const auto sources = filtered.sourceWeightPlan(directory, ModelContextId{52});
                ASSERT_EQ(sources.size(), 2u);
                for (const auto &source : sources.requirements())
                    EXPECT_EQ(source.role, WeightRole::MoEExpertDown);
                const auto whole = MoEExpertOverlayPreparationPlan::build(*wholeRuntime(*runtime), directory)
                    .filteredForDevice(participant.local_device).sourceWeightPlan(directory, ModelContextId{52});
                EXPECT_TRUE(whole.empty());
            }
        }
        EXPECT_EQ(empty_owners, 3u);
    }
}

/** @test Registry construction derives its immutable family from preparation, including empty owners. */
TEST(MoEProjectionPreparation, RegistryAuthenticatesEveryLocalLayerBeforePublication)
{
    const SourceDirectory directory({1, 512, 256});
    for (const bool rocm : {false, true})
    {
        auto source = planFor(rocm, 4, RoutedExpertOwnerOrder::Ordinal, 1);
        source->placements.resize(1); // This fixture owns one real layer, not a sparse layer catalog.
        const auto runtime = resolveMoEExpertOverlayRuntimePlan(source);
        const auto prep = std::make_shared<const MoEExpertOverlayPreparationPlan>(
            MoEExpertOverlayPreparationPlan::build(*runtime, directory));
        const auto owners = MoEExpertOwnerMap::build(*source);
        for (const auto &participant : owners.participants())
        {
            MoEOverlayParticipantResidencyRegistry::Config config{
                .owner_map = owners, .local_participant_ids = {participant.participant_id},
                .num_layers = 1, .num_experts = 1, .initial_epoch = 1, .projection_preparation = prep};
            MoEOverlayParticipantResidencyRegistry registry(config);
            auto endpoint = registry.endpoint(participant.participant_id);
            ASSERT_NE(endpoint, nullptr);
            EXPECT_EQ(endpoint->movableProjections(), DeviceMoEProjectionSet::GateUp);
            const auto layout = prep->requireProjectionOwnershipForParticipant(participant, 0);
            EXPECT_EQ(layout.participant(), participant.domain_participant_index);
            if (owners.expertsForParticipant(0, participant.participant_id).empty())
                EXPECT_EQ(endpoint->retainedEpochCount(), 1u);
            else
                EXPECT_EQ(endpoint->retainedEpochCount(), 0u);

            auto invalid = config;
            invalid.projection_preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
                MoEExpertOverlayPreparationPlan::build(*wholeRuntime(*runtime), directory));
            EXPECT_THROW((MoEOverlayParticipantResidencyRegistry{invalid}), std::invalid_argument);
            invalid = config;
            invalid.num_layers = 2;
            // Complete canonical ownership is authenticated before projection
            // geometry; a foreign interval is an ownership contract failure.
            EXPECT_THROW((MoEOverlayParticipantResidencyRegistry{invalid}), std::logic_error);
            invalid = config;
            invalid.num_experts = 2;
            EXPECT_THROW((MoEOverlayParticipantResidencyRegistry{invalid}), std::logic_error);
        }
    }
}

TEST(MoEProjectionPreparation, SourceSelectionsFollowEachRoleAndRetainFullHostAxes)
{
    const SourceDirectory directory(geometry);
    for (const bool rocm : {false, true})
        for (const int participants : {2, 3, 4, 8})
            for (const auto order : {RoutedExpertOwnerOrder::Ordinal, RoutedExpertOwnerOrder::Random})
            {
                const auto runtime = resolveMoEExpertOverlayRuntimePlan(planFor(rocm, participants, order));
                for (const auto &prep : {MoEExpertOverlayPreparationPlan::build(*wholeRuntime(*runtime), directory),
                     MoEExpertOverlayPreparationPlan::build(*runtime, directory)})
                {
                    const auto sources = prep.sourceWeightPlan(directory, ModelContextId{53});
                    EXPECT_EQ(sources.physicalMemoryOwner(), PhysicalMemoryOwner::RoutedExpertWeights);
                    EXPECT_EQ(sources.strategy().model_id, ModelContextId{53});
                    EXPECT_EQ(sources.strategy().devices.size(), static_cast<size_t>(participants));
                    EXPECT_EQ(sources.size(), static_cast<size_t>(6 * participants));
                    size_t projection_count = 0;
                    for (const auto &source : sources.requirements())
                    {
                        SCOPED_TRACE(source.canonical_name + "/" + source.target_device.to_string());
                        const auto native_shape = *directory.getTensorShape(source.canonical_name);
                        EXPECT_EQ(source.slice.row_count, native_shape[0]);
                        EXPECT_EQ(source.slice.col_count, native_shape[1]);
                        EXPECT_EQ(source.slice.source_rows, native_shape[0]);
                        EXPECT_EQ(source.slice.source_cols, native_shape[1]);
                        EXPECT_EQ(source.slice.row_start, 0u);
                        EXPECT_EQ(source.slice.col_start, 0u);
                        EXPECT_TRUE(source.slice.inner_is_presliced);
                        EXPECT_EQ(source.derivation, WeightDerivationKind::ExpertSlice);
                        EXPECT_EQ(source.host_policy, WeightHostPolicy::RequiredUntilPreparedOrTransferred);
                        EXPECT_EQ(source.residency_category, WeightResidencyCategory::AcceleratorRoutedExpert);
                        EXPECT_EQ(source.lookup_device, DeviceId::cpu());
                        EXPECT_TRUE(source.bypass_tensor_parallel);
                        const auto role = source.role == WeightRole::MoEExpertGate ? Role::GATE :
                            source.role == WeightRole::MoEExpertUp ? Role::UP : Role::DOWN;
                        const auto ids = prep.expertsForDomainDeviceLayerRole(
                            source.overlay_domain, source.target_device, source.layer, role);
                        EXPECT_EQ(source.slice.expert_ids, ids);
                        EXPECT_EQ(source.slice.expert_start, static_cast<size_t>(ids.front()));
                        EXPECT_EQ(source.slice.expert_count, ids.size());
                        for (int expert : ids)
                            EXPECT_NE(prep.requestForParticipant(source.overlay_domain, source.target_device,
                                source.overlay_participant_world_rank, source.overlay_participant_index,
                                source.layer, expert, role), nullptr);
                        projection_count += ids.size();
                    }
                    EXPECT_EQ(projection_count, prep.requests().size());
                }
            }
}

TEST(MoEProjectionPreparation, SourcePlanRejectsStaleOrMissingGeometryBeforeLoading)
{
    const SourceDirectory valid(geometry);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(planFor(true, 2, RoutedExpertOwnerOrder::Random));
    const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime, valid);
    for (const auto &shape : std::vector<std::vector<size_t>>{
        {}, {256, 1536}, {256, 0, 24}, {256, 1536, 23}, {1536, 256, 24}, {256, 768, 24}})
    {
        SourceDirectory directory(geometry);
        directory.shapes["blk.3.ffn_down_exps.weight"] = shape;
        EXPECT_THROW(prep.sourceWeightPlan(directory, ModelContextId{54}), std::invalid_argument);
        EXPECT_THROW(MoEExpertOverlayPreparationPlan::build(*runtime, directory),
            std::invalid_argument);
    }
    SourceDirectory missing(geometry);
    missing.shapes.erase("blk.0.ffn_up_exps.weight");
    EXPECT_THROW(prep.sourceWeightPlan(missing, ModelContextId{54}), std::invalid_argument);
    SourceDirectory fewer_experts({23, 1536, 256});
    EXPECT_THROW(MoEExpertOverlayPreparationPlan::build(*wholeRuntime(*runtime), valid)
        .sourceWeightPlan(fewer_experts, ModelContextId{54}), std::invalid_argument);
}

TEST(MoEProjectionPreparation, RejectsUnprovenGeometryAndPhysicalLayouts)
{
    const SourceDirectory directory(geometry);
    const auto plan = planFor(true, 2, RoutedExpertOwnerOrder::Ordinal);
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(plan);
    for (const auto invalid : {MoEExpertProjectionOwnership::Geometry{23, 1536, 256},
         MoEExpertProjectionOwnership::Geometry{24, 1535, 256}, MoEExpertProjectionOwnership::Geometry{24, 1536, 0}})
        EXPECT_THROW(MoEExpertOverlayPreparationPlan::build(*runtime, SourceDirectory(invalid)),
            std::invalid_argument);

    auto replicated = planFor(true, 2, RoutedExpertOwnerOrder::Ordinal);
    replicated->domains.front().routed_compute_policy = RoutedExpertComputePolicy::Replicated;
    const auto replicated_runtime = resolveMoEExpertOverlayRuntimePlan(replicated);
    const auto replicated_prep = MoEExpertOverlayPreparationPlan::build(*replicated_runtime, directory);
    for (const auto &request : replicated_prep.requests()) EXPECT_FALSE(request.projection_ownership);

    auto cpu = planFor(true, 2, RoutedExpertOwnerOrder::Ordinal);
    cpu->domains.front().participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
    cpu->domains.front().backend = CollectiveBackendType::UPI;
    cpu->domains.front().scope = ExecutionDomainScope::NODE_LOCAL;
    cpu->domains.front().world_ranks = {0, 1};
    // Unsupported CPU execution is rejected at admission, before preparation
    // or model mapping. A test must not require the failure to happen later.
    EXPECT_THROW(resolveMoEExpertOverlayRuntimePlan(cpu), std::invalid_argument);
}

TEST(MoEProjectionPreparation, UsesEachLayersOwnSourceGeometry)
{
    SourceDirectory directory(geometry);
    directory.shapes["blk.3.ffn_gate_exps.weight"] = {1536, 512, 24};
    directory.shapes["blk.3.ffn_up_exps.weight"] = {1536, 512, 24};
    directory.shapes["blk.3.ffn_down_exps.weight"] = {512, 1536, 24};
    const auto runtime = resolveMoEExpertOverlayRuntimePlan(planFor(true, 3, RoutedExpertOwnerOrder::Random));
    const auto prep = MoEExpertOverlayPreparationPlan::build(*runtime, directory);
    for (const auto &request : prep.requests())
    {
        ASSERT_TRUE(request.projection_ownership);
        EXPECT_EQ(request.projection_ownership->geometry().intermediate_columns, request.layer == 0 ? 256 : 512);
    }
    const auto sources = prep.sourceWeightPlan(directory, ModelContextId{55});
    ASSERT_EQ(sources.size(), 18u);
    for (const auto &source : sources.requirements())
        EXPECT_EQ(source.role == WeightRole::MoEExpertDown ? source.slice.source_rows : source.slice.source_cols,
            source.layer == 0 ? 256u : 512u);
}

TEST(MoEProjectionPreparation, CpuSourcesRetainRankIdentityAndHostExecutionLifetime)
{
    auto plan = planFor(true, 2, RoutedExpertOwnerOrder::Random);
    auto &domain = plan->domains.front();
    domain.participants = {GlobalDeviceAddress::cpu(0, "bound-node"), GlobalDeviceAddress::cpu(1, "bound-node")};
    domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
    domain.backend = CollectiveBackendType::UPI;
    domain.scope = ExecutionDomainScope::NODE_LOCAL;
    domain.world_ranks = {0, 1};
    plan->continuation_domain_spec.setDensePolicy(DenseParallelPolicy::TensorParallel);
    const SourceDirectory directory(geometry);
    const auto execution = resolveMoEExpertOverlayExecutionPlan(plan,
        MoEExpertOverlayExecutionPlanResolverOptions{.current_world_rank = 0, .world_size = 2});
    for (int rank = 0; rank < 2; ++rank)
    {
        const auto runtime = resolveMoEExpertOverlayRuntimePlan(plan,
            MoEExpertOverlayRuntimeResolverOptions{.current_world_rank = rank, .validate_mvp_root_reachability = false});
        const auto preparation = MoEExpertOverlayPreparationPlan::build(*runtime)
            .filteredForRank(*execution.rankPlanFor(rank));
        const auto sources = preparation.sourceWeightPlan(directory, ModelContextId{56});
        ASSERT_EQ(sources.size(), 6u);
        for (const auto &source : sources.requirements())
        {
            EXPECT_TRUE(source.target_device.is_cpu());
            EXPECT_EQ(source.overlay_participant_world_rank, rank);
            EXPECT_EQ(source.overlay_participant_index, rank);
            EXPECT_EQ(source.slice.expert_count, 12u);
            EXPECT_EQ(source.host_policy, WeightHostPolicy::RequiredForCPUExecution);
            EXPECT_EQ(source.residency_category, WeightResidencyCategory::CpuFallbackExpert);
        }
    }
}
} // namespace llaminar2::test
