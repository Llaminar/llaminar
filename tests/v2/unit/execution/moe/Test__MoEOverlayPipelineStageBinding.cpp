/**
 * @file Test__MoEOverlayPipelineStageBinding.cpp
 * @brief Device-free regressions for the production PP-to-child expert handoff.
 *
 * Real residency authorities and endpoint registries carry tiny model metadata;
 * no device or model payload is loaded. Tests call the same RankOrchestrator
 * projection used by both nested TP and single-device PP construction, including
 * foreign scopes, endpoint substitutions, mode changes and terminal MTP banks.
 * CPU root declarations remain stage-owned across child projection and reuse.
 */
#include "execution/moe/MoEOverlayPipelineStageBinding.h"
#include "memory/CPUSerialExpertStorageLayout.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/moe/MoEOverlayRankBatchTransport.h"
#include "../../../mocks/MockModelContext.h"
#include "../../../mocks/MockRankOrchestrator.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        using Config = RankOrchestrator::Config;
        using Stage = RankOrchestrator::PPStageConfig;
        using Binding = MoEOverlayPipelineStageBinding;

        /** @brief Real, compact authority and empty prepared endpoints for one stage. */
        struct StageFixture
        {
            FactoryPPStageConfig scope;
            MoERoutedExpertModelMetadata metadata;
            std::vector<GlobalDeviceAddress> devices;
            CollectiveBackendType backend;
            std::vector<float> workshare;
            Binding::Owners owners;

            /**
             * @brief Construct stage geometry without touching an accelerator.
             * @param first Global origin of two owned main layers.
             * @param terminal Whether the stage owns the model's output boundary.
             * @param auxiliary Retained routed NextN banks, only valid at the terminal.
             * @param type Device family used as immutable topology metadata.
             * @param width Single-device PP or multi-device local TP.
             * @param mode Off, Observe and Dynamic retain distinct explicit policies.
             * @param first_ordinal First physical ordinal, keeping sibling stages distinct.
             * @param weights Authored TP declaration, including explicit singleton workshare.
             */
            StageFixture(int first, bool terminal, int auxiliary, DeviceType type,
                         int width, MoERebalanceRuntimeMode mode, int first_ordinal = 0,
                         std::vector<float> weights = {})
                : scope{.first_layer = first, .last_layer = first + 2,
                        .has_embedding = first == 0, .has_lm_head = terminal},
                  backend(type == DeviceType::CUDA ? CollectiveBackendType::NCCL :
                          type == DeviceType::ROCm ? CollectiveBackendType::RCCL : CollectiveBackendType::HOST),
                  workshare(std::move(weights))
            {
                for (int ordinal = 0; ordinal < width; ++ordinal)
                    devices.push_back(type == DeviceType::CUDA ? GlobalDeviceAddress::cuda(first_ordinal + ordinal, 0) :
                        type == DeviceType::ROCm ? GlobalDeviceAddress::rocm(first_ordinal + ordinal, 0) :
                        GlobalDeviceAddress::cpu(first_ordinal + ordinal));
                metadata.num_layers = 2 + auxiliary;
                metadata.num_experts = 4;
                metadata.d_model = 8;
                metadata.routed_intermediate_size = 16;
                metadata.first_model_layer = first;
                metadata.main_inference_layer_count = first + 2;
                MoERoutedExpertPlacementPlan plan;
                plan.enabled = true;
                plan.topology = RoutedExpertPlacementTopology::SingleDomain;
                plan.continuation_domain = "stage";
                plan.base_model_domain = "stage";
                plan.shared_expert_domain = "stage";
                plan.first_model_layer = first;
                plan.residency_policy = defaultRoutedExpertResidencyPolicy(mode);
                RoutedExpertDomain domain;
                domain.name = "stage";
                domain.scope = width == 1 ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::RANK_LOCAL;
                domain.backend = backend;
                domain.participants = devices;
                domain.weights = workshare;
                domain.owner_rank = 0;
                domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
                plan.domains.push_back(domain);
                RoutedExpertTier tier;
                tier.name = "all";
                tier.domain = "stage";
                tier.priority = 0;
                tier.fallback = true;
                plan.routed_tiers.push_back(tier);
                plan.continuation_domain_spec.domain = "stage";
                plan.continuation_domain_spec.setDensePolicy(
                    defaultMoEContinuationDensePolicy(domain.scope, devices.size()));
                plan.authority_execution = resolveMoEOverlayAuthorityExecutionKind(plan);
                if (mode != MoERebalanceRuntimeMode::Off)
                {
                    DecodeExpertHistogramConfig histogram;
                    histogram.num_layers = metadata.num_layers;
                    histogram.num_experts = metadata.num_experts;
                    histogram.top_k = 1;
                    histogram.window_size = 4;
                    histogram.token_boundary_layer_idx = scope.last_layer - 1;
                    std::vector<int> expert_owners(4);
                    for (int expert = 0; expert < 4; ++expert) expert_owners[expert] = expert % width;
                    histogram.ownership = MoELayeredExpertOwnership::uniform(
                        metadata.num_layers, width, expert_owners, first);
                    for (const auto &device : devices) histogram.sockets.push_back(device.toLocalDeviceId());
                    owners.histogram = std::make_shared<DecodeExpertHistogram>(std::move(histogram));
                }
                owners.authority = std::make_shared<MoEOverlayResidencyAuthority>(
                    MoEOverlayResidencyAuthority::Config{.initial_plan = plan,
                        .model_metadata = metadata, .maintenance_mode = mode,
                        .histogram = owners.histogram.get()});
                const auto snapshot = owners.authority->snapshot();
                std::vector<int> local_ids(width);
                std::iota(local_ids.begin(), local_ids.end(), 0);
                owners.residency = std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                    MoEOverlayParticipantResidencyRegistry::Config{
                        .owner_map = snapshot->owner_map, .local_participant_ids = local_ids,
                        .num_layers = metadata.num_layers, .num_experts = metadata.num_experts,
                        .initial_epoch = snapshot->epoch, .first_model_layer = first});
            }

            /** @return Production-checked runtime handoff retaining this fixture's owners. */
            std::shared_ptr<const Binding> seal() const
            { return Binding::seal(scope, metadata, devices, owners); }

            /** @return Authored PP stage with an optional completed runtime installation. */
            Stage stage(bool install = true) const
            {
                Stage stage{.first_layer = scope.first_layer, .last_layer = scope.last_layer,
                    .has_embedding = scope.has_embedding, .has_lm_head = scope.has_lm_head,
                    .stage_devices = devices, .tp_weights = workshare, .tp_backend = backend};
                if (install) stage.moe_runtime = seal();
                return stage;
            }
        };

        /**
         * @brief Author the complete parent boundary before sealing either fixture.
         * @param first Initial stage's immutable participants and scope.
         * @param last Terminal stage's immutable participants and scope.
         */
        void declarePipelineGenerationDomain(StageFixture &first, StageFixture &last)
        {
            DeviceGenerationDomainDeclaration declaration;
            for (const auto &member : first.devices) declaration.observe(member.device_type);
            for (const auto &member : last.devices) declaration.observe(member.device_type);
            first.scope.generation_domain = declaration.domain();
            last.scope.generation_domain = declaration.domain();
        }

        /** @return Dense two-stage policy used to detect accidental inheritance. */
        Config densePipeline()
        {
            Config result;
            result.mode = RankOrchestrator::ParallelismMode::PP;
            result.max_seq_len = 262144;
            result.resident_graph_rows = 128;
            result.batch_size = 2;
            result.pp_stages = {
                {.first_layer = 0, .last_layer = 2, .has_embedding = true,
                 .stage_devices = {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)},
                 .tp_backend = CollectiveBackendType::RCCL},
                {.first_layer = 2, .last_layer = 4, .has_lm_head = true,
                 .stage_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                 .tp_backend = CollectiveBackendType::NCCL}};
            return result;
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RetainsExactScopeAcrossDevicesWidthsAndModes)
    {
        for (const auto type : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const int width : {1, 2, 4, 8})
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe,
                               MoERebalanceRuntimeMode::Dynamic})
        for (const int first : {0, 32, 40})
        for (const bool terminal : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << " width=" << width
                << " mode=" << static_cast<int>(mode) << " first=" << first << " terminal=" << terminal);
            StageFixture fixture(first, terminal, terminal ? 1 : 0, type, width, mode);
            const auto binding = fixture.seal();
            ASSERT_TRUE(binding);
            EXPECT_EQ(binding->plan()->first_model_layer, first);
            EXPECT_EQ(binding->owners().authority, fixture.owners.authority);
            EXPECT_EQ(binding->owners().residency, fixture.owners.residency);
            EXPECT_EQ(binding->owners().histogram, fixture.owners.histogram);
            EXPECT_NO_THROW(binding->requireDestination(fixture.scope, fixture.devices, {}, fixture.backend, mode));
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, NoPeerKeepsDynamicAuthorityAndZeroMovementStorage)
    {
        for (const auto type : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const int first : {0, 24, 40})
        {
            StageFixture fixture(first, true, 1, type, 1, MoERebalanceRuntimeMode::Dynamic);
            const auto binding = fixture.seal();
            const auto &proof = binding->owners().authority->noPeerResidency();
            ASSERT_TRUE(proof);
            EXPECT_EQ(binding->owners().authority->maintenanceMode(), MoERebalanceRuntimeMode::Dynamic);
            EXPECT_EQ(proof->authorityExecution(), binding->plan()->authority_execution);
            EXPECT_NO_THROW(proof->requireDestination(fixture.devices.front().toLocalDeviceId(),
                fixture.metadata.first_model_layer, fixture.metadata.num_layers, fixture.metadata.num_experts));
            const auto storage = MoEOverlayCapacityAdmission::selectMigrationStorage(
                *binding->plan(), MoERebalanceRuntimeMode::Dynamic);
            EXPECT_EQ(storage, MoEOverlayMigrationStorageKind::NoPeer);
            const MoEOverlayCapacityAdmissionPolicy capacity{.migration_storage = storage};
            EXPECT_FALSE(capacity.migrationEnabled());
            EXPECT_FALSE(capacity.usesPhysicalResidencyFabric());
            EXPECT_FALSE(capacity.usesDeviceTransferDirectory());
            EXPECT_EQ(capacity.shadow_slots_per_endpoint_layer, 0u);
            EXPECT_EQ(capacity.staging_capacity_bytes, 0u);
            EXPECT_EQ(proof->status().completed_movement.physical_bytes, 0u);
            EXPECT_EQ(proof->status().completed_decision_windows, 0u);
            EXPECT_THROW(proof->requireDestination(fixture.devices.front().toLocalDeviceId(),
                fixture.metadata.first_model_layer + 1, fixture.metadata.num_layers, fixture.metadata.num_experts),
                std::invalid_argument);
        }
    }

    /** @test Native scalar tail ownership rejects absent, movable and foreign proofs. */
    TEST(Test__MoEOverlayPipelineStageBinding, NativePipelineSamplerRequiresExactNoPeerTail)
    {
        for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe,
                                MoERebalanceRuntimeMode::Dynamic})
        for (const int retained_sidecars : {0, 1})
        {
            StageFixture fixture(24, true, retained_sidecars, type, 1, mode);
            const auto &proof = fixture.owners.authority->noPeerResidency();
            const auto device = fixture.devices.front().toLocalDeviceId();
            EXPECT_NO_THROW(MoEOverlayNoPeerResidency::requireNativePipelineTail(
                proof, fixture.scope, device, fixture.metadata.num_experts));
            EXPECT_THROW(MoEOverlayNoPeerResidency::requireNativePipelineTail(
                std::nullopt, fixture.scope, device, fixture.metadata.num_experts), std::invalid_argument);
            for (int defect = 0; defect < 8; ++defect)
            {
                auto stage = fixture.scope;
                auto endpoint = device;
                int experts = fixture.metadata.num_experts;
                if (defect == 0) stage.has_lm_head = false;
                if (defect == 1) stage.has_embedding = true;
                if (defect == 2) stage.first_layer++;
                if (defect == 3) stage.last_layer = stage.first_layer + fixture.metadata.num_layers + 1;
                if (defect == 4) endpoint.ordinal++;
                if (defect == 5) experts++;
                if (defect == 6) stage.last_layer--; // Still nonempty: a truncated main interval.
                if (defect == 7) stage.last_layer++; // Fits routed capacity when a sidecar is retained.
                EXPECT_THROW(MoEOverlayNoPeerResidency::requireNativePipelineTail(
                    proof, stage, endpoint, experts), std::invalid_argument);
            }
            const auto snapshot = fixture.owners.authority->snapshot();
            auto unresolved = fixture.metadata;
            unresolved.main_inference_layer_count = 0;
            const auto construction_only = MoEOverlayNoPeerResidency::seal(
                *snapshot->placement_plan, snapshot->owner_map, unresolved);
            ASSERT_TRUE(construction_only);
            EXPECT_THROW(MoEOverlayNoPeerResidency::requireNativePipelineTail(
                construction_only, fixture.scope, device, fixture.metadata.num_experts), std::invalid_argument);
            for (const int invalid_end : {-1, fixture.metadata.first_model_layer,
                    fixture.metadata.first_model_layer + fixture.metadata.num_layers + 1})
            {
                auto malformed = fixture.metadata;
                malformed.main_inference_layer_count = invalid_end;
                EXPECT_THROW((void)MoEOverlayNoPeerResidency::seal(
                    *snapshot->placement_plan, snapshot->owner_map, malformed), std::invalid_argument);
            }
            StageFixture movable(24, true, retained_sidecars, type, 2, mode);
            EXPECT_FALSE(movable.owners.authority->noPeerResidency());
            EXPECT_THROW(MoEOverlayNoPeerResidency::requireNativePipelineTail(
                movable.owners.authority->noPeerResidency(), movable.scope, device,
                movable.metadata.num_experts), std::invalid_argument);
        }
        StageFixture host(24, true, 0, DeviceType::CPU, 1, MoERebalanceRuntimeMode::Dynamic);
        EXPECT_THROW(MoEOverlayNoPeerResidency::requireNativePipelineTail(
            host.owners.authority->noPeerResidency(), host.scope, DeviceId::cpu(),
            host.metadata.num_experts), std::invalid_argument);
    }

    TEST(Test__MoEOverlayPipelineStageBinding, NoPeerRejectsMissingForeignOrNonresidentMembership)
    {
        StageFixture fixture(24, true, 1, DeviceType::CPU, 1, MoERebalanceRuntimeMode::Dynamic);
        const auto snapshot = fixture.owners.authority->snapshot();
        for (int mutation = 0; mutation < 7; ++mutation)
        {
            SCOPED_TRACE(mutation);
            auto plan = *snapshot->placement_plan;
            switch (mutation)
            {
            case 0: plan.placements.pop_back(); break;
            case 1: plan.placements.front().routed_expert_tier.pop_back(); break;
            case 2: plan.placements.front().routed_expert_tier.front() = -1; break;
            case 3: plan.placements.back().layer = plan.placements.front().layer; break;
            case 4: plan.authority_execution = MoEOverlayAuthorityExecutionKind::DeviceResident; break;
            case 5: plan.domains.front().owner_rank = 1; break;
            case 6: plan.first_model_layer = 0; break;
            }
            EXPECT_THROW((void)MoEOverlayNoPeerResidency::seal(plan, snapshot->owner_map, fixture.metadata),
                std::invalid_argument);
        }
        StageFixture movable(24, true, 1, DeviceType::CPU, 2, MoERebalanceRuntimeMode::Dynamic);
        EXPECT_FALSE(movable.owners.authority->noPeerResidency());
        EXPECT_EQ(MoEOverlayCapacityAdmission::selectMigrationStorage(*movable.seal()->plan(),
            MoERebalanceRuntimeMode::Dynamic), MoEOverlayMigrationStorageKind::PhysicalResidencyFabric);
        StageFixture native(24, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
        EXPECT_FALSE(native.owners.authority->noPeerResidency());
        EXPECT_EQ(MoEOverlayCapacityAdmission::selectMigrationStorage(*native.seal()->plan(),
            MoERebalanceRuntimeMode::Dynamic), MoEOverlayMigrationStorageKind::DeviceTransferDirectory);
    }

    /** @brief Device-free participant for the actual Rank ownership validator. */
    class AuthorityWitness final : public MockDeviceRunner
    {
    public:
        /** @brief Retain the exact immutable execution identity; never initialize a backend. */
        AuthorityWitness(DeviceId device, MoEOverlayAuthorityExecutionKind authority)
            : device_(device), authority_(authority) {}
        /** @return Explicit simulated participant-local device identity. */
        DeviceId primaryDeviceId() const override { return device_; }
        /** @return Exact child authority; mutable only for this adversarial test. */
        MoEOverlayAuthorityExecutionKind moeOverlayAuthorityExecution() const override { return authority_; }
        /** @return No fake maintenance executable is admitted by this witness. */
        bool deviceResidentMoEOverlayMaintenanceReady() const override { return maintenance_ready_; }
        /** @return The same graph-lifecycle kind as the declared participant. */
        ServingGraphPreparationKind servingGraphPreparationKind() const noexcept override
        { return device_.is_gpu() ? ServingGraphPreparationKind::NativeDeviceExecutableFamily
                                 : ServingGraphPreparationKind::EagerHostGraph; }
        DeviceId device_;
        MoEOverlayAuthorityExecutionKind authority_;
        bool maintenance_ready_ = false;
    };

    TEST(Test__MoEOverlayPipelineStageBinding, RankAcceptsIndependentNoPeerLociAndRejectsLaterDisagreement)
    {
        StageFixture first(0, false, 0, DeviceType::CUDA, 1, MoERebalanceRuntimeMode::Dynamic);
        StageFixture last(2, true, 1, DeviceType::CPU, 1, MoERebalanceRuntimeMode::Dynamic);
        declarePipelineGenerationDomain(first, last);
        auto config = densePipeline();
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        config.pp_stages = {first.stage(), last.stage()};
        auto head = std::make_unique<AuthorityWitness>(DeviceId::cuda(0), MoEOverlayAuthorityExecutionKind::DeviceResident);
        auto tail = std::make_unique<AuthorityWitness>(DeviceId::cpu(), MoEOverlayAuthorityExecutionKind::HostResident);
        auto *tail_ptr = tail.get();
        std::vector<std::unique_ptr<IInferenceRunner>> children;
        children.push_back(std::move(head)); children.push_back(std::move(tail));
        auto rank = RankOrchestrator::createForTestWithPipelineStages(MockModelContext::createMinimal(),
            std::move(children), config);
        using Frontier = IInferenceRunner::MoEPipelineOwnershipValidation;
        EXPECT_NO_THROW(rank->requireMoEPipelineRuntimeOwnership(Frontier::OwnersBound));
        EXPECT_THROW(rank->requireMoEPipelineRuntimeOwnership(static_cast<Frontier>(255)), std::invalid_argument);
        EXPECT_THROW((void)rank->moeOverlayAuthorityExecution(), std::logic_error)
            << "One overlay's singular query must still reject unrelated stage authorities";
        tail_ptr->authority_ = MoEOverlayAuthorityExecutionKind::DeviceResident;
        EXPECT_THROW(rank->requireMoEPipelineRuntimeOwnership(Frontier::OwnersBound), std::logic_error);
        EXPECT_FALSE(first.owners.residency->allInitialBanksReady());
        EXPECT_FALSE(last.owners.residency->allInitialBanksReady());
        tail_ptr->authority_ = MoEOverlayAuthorityExecutionKind::HostResident;
        tail_ptr->device_ = DeviceId::rocm(0);
        EXPECT_THROW(rank->requireMoEPipelineRuntimeOwnership(Frontier::OwnersBound), std::logic_error);
        tail_ptr->device_ = DeviceId::cpu();
        EXPECT_NO_THROW(rank->requireMoEPipelineRuntimeOwnership(Frontier::OwnersBound));
        EXPECT_THROW(rank->requireMoEPipelineRuntimeOwnership(Frontier::ServingMaterialized), std::logic_error)
            << "Immutable membership cannot certify missing physical prepared banks";
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RankDoesNotExemptMovableMissingMaintenance)
    {
        StageFixture first(0, false, 0, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
        StageFixture last(2, true, 1, DeviceType::CPU, 1, MoERebalanceRuntimeMode::Dynamic);
        declarePipelineGenerationDomain(first, last);
        auto config = densePipeline();
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        config.pp_stages = {first.stage(), last.stage()};
        MockLocalTPContext::Config context_config;
        context_config.devices = first.devices;
        context_config.backend = CollectiveBackendType::NCCL;
        auto context = std::make_unique<MockLocalTPContext>(context_config);
        std::vector<std::unique_ptr<IInferenceRunner>> members;
        members.push_back(std::make_unique<AuthorityWitness>(DeviceId::cuda(0), MoEOverlayAuthorityExecutionKind::DeviceResident));
        members.push_back(std::make_unique<AuthorityWitness>(DeviceId::cuda(1), MoEOverlayAuthorityExecutionKind::DeviceResident));
        auto domain = RankOrchestrator::createForTest(MockModelContext::createMinimal(), std::move(members),
            std::move(context), config.forPipelineStage(0));
        std::vector<std::unique_ptr<IInferenceRunner>> children;
        children.push_back(std::move(domain));
        children.push_back(std::make_unique<AuthorityWitness>(DeviceId::cpu(), MoEOverlayAuthorityExecutionKind::HostResident));
        auto rank = RankOrchestrator::createForTestWithPipelineStages(MockModelContext::createMinimal(),
            std::move(children), config);
        using Frontier = IInferenceRunner::MoEPipelineOwnershipValidation;
        EXPECT_NO_THROW(rank->requireMoEPipelineRuntimeOwnership(Frontier::OwnersBound));
        EXPECT_THROW(rank->requireMoEPipelineRuntimeOwnership(Frontier::ServingMaterialized), std::logic_error);
        EXPECT_FALSE(first.owners.authority->noPeerResidency());
        EXPECT_EQ(first.owners.authority->maintenanceMode(), MoERebalanceRuntimeMode::Dynamic);
        EXPECT_FALSE(first.owners.residency->allInitialBanksReady());
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsForeignOrMissingRuntimeOwners)
    {
        for (int mutation = 0; mutation < 8; ++mutation)
        {
            SCOPED_TRACE(mutation);
            StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
            StageFixture other(0, false, 0, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Dynamic);
            auto owners = fixture.owners;
            switch (mutation)
            {
            case 0: owners.authority.reset(); break;
            case 1: owners.residency.reset(); break;
            case 2: owners.histogram.reset(); break;
            case 3: owners.authority = other.owners.authority; break;
            case 4: owners.residency = other.owners.residency; break;
            case 5: owners.histogram = other.owners.histogram; break;
            case 6: {
                StageFixture foreign_device(32, true, 1, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Dynamic);
                owners.residency = foreign_device.owners.residency; break;
            }
            case 7: {
                StageFixture wrong_width(32, true, 1, DeviceType::CUDA, 1, MoERebalanceRuntimeMode::Dynamic);
                owners.residency = wrong_width.owners.residency; break;
            }
            }
            EXPECT_ANY_THROW((void)Binding::seal(fixture.scope, fixture.metadata, fixture.devices, owners));
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsForeignGeometryAndNonterminalMTP)
    {
        for (int mutation = 0; mutation < 8; ++mutation)
        {
            SCOPED_TRACE(mutation);
            StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Off);
            auto metadata = fixture.metadata;
            auto scope = fixture.scope;
            switch (mutation)
            {
            case 0: metadata.first_model_layer = 0; break;
            case 1: metadata.num_layers = 1; break;
            case 2: metadata.num_experts = 5; break;
            case 3: metadata.main_inference_layer_count = 33; break;
            case 4: metadata.num_layers = std::numeric_limits<int>::max(); break;
            case 5: scope.has_lm_head = false; break;
            case 6: scope.has_embedding = true; break;
            case 7: scope.first_layer = -1; break;
            }
            EXPECT_ANY_THROW((void)Binding::seal(scope, metadata, fixture.devices, fixture.owners));
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsDifferentOwnerMapWithIdenticalEndpoints)
    {
        StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Off);
        auto owners = fixture.owners;
        const auto snapshot = owners.authority->snapshot();
        auto different = snapshot->layered_ownership;
        for (int layer = 32; layer < 35; ++layer)
            for (int expert = 0; expert < 4; ++expert)
                different.assignOwner(layer, expert, 1 - different.owner(layer, expert));
        owners.residency = std::make_shared<MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{
                .owner_map = MoEExpertOwnerMap::buildExplicit(*snapshot->placement_plan, different),
                .local_participant_ids = {0, 1}, .num_layers = 3, .num_experts = 4,
                .initial_epoch = snapshot->epoch, .first_model_layer = 32});
        EXPECT_THROW((void)Binding::seal(fixture.scope, fixture.metadata, fixture.devices, owners),
                     std::invalid_argument);
    }

    TEST(Test__MoEOverlayPipelineStageBinding, RejectsChangedDestinationAndTPDeclaration)
    {
        StageFixture fixture(32, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
        const auto binding = fixture.seal();
        for (int mutation = 0; mutation < 8; ++mutation)
        {
            SCOPED_TRACE(mutation);
            auto scope = fixture.scope;
            auto devices = fixture.devices;
            auto backend = fixture.backend;
            auto mode = MoERebalanceRuntimeMode::Dynamic;
            std::vector<float> weights;
            switch (mutation)
            {
            case 0: scope.first_layer = 0; break;
            case 1: scope.last_layer = 33; break;
            case 2: scope.has_lm_head = false; break;
            case 3: std::reverse(devices.begin(), devices.end()); break;
            case 4: devices.pop_back(); break;
            case 5: backend = CollectiveBackendType::HOST; break;
            case 6: weights = {0.25f, 0.75f}; break;
            case 7: mode = MoERebalanceRuntimeMode::Off; break;
            }
            EXPECT_THROW(binding->requireDestination(scope, devices, weights, backend, mode), std::invalid_argument);
        }
    }

    /** @test Actual plan translation preserves every authored singleton and grouped TP declaration. */
    TEST(Test__MoEOverlayPipelineStageBinding, FromPlanPreservesSingletonAndGroupedDeclarations)
    {
        for (const auto types : {std::array{DeviceType::CUDA, DeviceType::CUDA},
                                std::array{DeviceType::ROCm, DeviceType::ROCm},
                                std::array{DeviceType::CPU, DeviceType::CPU},
                                std::array{DeviceType::CUDA, DeviceType::ROCm},
                                std::array{DeviceType::ROCm, DeviceType::CUDA}})
        for (const int width : {1, 2, 4})
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe,
                               MoERebalanceRuntimeMode::Dynamic})
        for (const bool explicit_weights : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "first=" << static_cast<int>(types[0])
                << " last=" << static_cast<int>(types[1]) << " width=" << width
                << " mode=" << static_cast<int>(mode) << " explicit_weights=" << explicit_weights);
            const auto weights = explicit_weights ? std::vector<float>(width, 1.0f / width) : std::vector<float>{};
            StageFixture early(0, false, 0, types[0], width, mode, 0, weights);
            StageFixture terminal(2, true, 0, types[1], width, mode, width, weights);
            declarePipelineGenerationDomain(early, terminal);
            RankExecutionPlan plan;
            plan.runtime.moe_rebalance.mode = mode;
            plan.local_pp_devices = {early.devices.front(), terminal.devices.front()};
            plan.local_pp_layer_boundaries = {0, 2, 4};
            plan.local_pp_stage_tp_info = {
                {.devices = early.devices, .tp_weights = weights, .tp_backend = early.backend},
                {.devices = terminal.devices, .tp_weights = weights, .tp_backend = terminal.backend}};
            auto parent = Config::fromPlan(plan);
            ASSERT_EQ(parent.pp_stages.size(), 2u);
            parent.pp_stages[0].moe_runtime = early.seal();
            parent.pp_stages[1].moe_runtime = terminal.seal();
            EXPECT_EQ(parent.pp_stages[0].tp_backend, early.backend);
            EXPECT_EQ(parent.pp_stages[1].tp_backend, terminal.backend);
            EXPECT_EQ(parent.pp_stages[0].tp_weights, weights);
            EXPECT_EQ(parent.pp_stages[1].tp_weights, weights);
            ASSERT_TRUE(parent.validate());
            for (std::size_t index = 0; index < 2; ++index)
            {
                const auto &fixture = index == 0 ? early : terminal;
                const auto child = parent.forPipelineStage(index);
                EXPECT_TRUE(child.validate());
                EXPECT_EQ(child.devices, fixture.devices);
                EXPECT_EQ(child.weights, weights);
                EXPECT_EQ(child.backend, fixture.backend);
                EXPECT_EQ(child.moe_expert_overlay_residency_authority, fixture.owners.authority);
                ASSERT_TRUE(child.nested_pp_stage_config);
                EXPECT_EQ(child.nested_pp_stage_config->first_layer, fixture.scope.first_layer);
                EXPECT_EQ(child.nested_pp_stage_config->last_layer, fixture.scope.last_layer);
                EXPECT_EQ(child.nested_pp_stage_config->has_embedding, fixture.scope.has_embedding);
                EXPECT_EQ(child.nested_pp_stage_config->has_lm_head, fixture.scope.has_lm_head);
            }
        }
    }

    /** @test Preserving a singleton declaration never exempts foreign scope, workshare or authority. */
    TEST(Test__MoEOverlayPipelineStageBinding, FromPlanSingletonStillRejectsForeignDeclaration)
    {
        for (const auto type : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        {
            StageFixture early(0, false, 0, type, 1, MoERebalanceRuntimeMode::Off);
            StageFixture terminal(2, true, 0, type, 1, MoERebalanceRuntimeMode::Off, 1);
            declarePipelineGenerationDomain(early, terminal);
            RankExecutionPlan plan;
            plan.runtime.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
            plan.local_pp_devices = {early.devices.front(), terminal.devices.front()};
            plan.local_pp_layer_boundaries = {0, 2, 4};
            plan.local_pp_stage_tp_info = {
                {.devices = early.devices, .tp_backend = early.backend},
                {.devices = terminal.devices, .tp_backend = terminal.backend}};
            auto parent = Config::fromPlan(plan);
            parent.pp_stages[0].moe_runtime = early.seal();
            parent.pp_stages[1].moe_runtime = terminal.seal();
            ASSERT_TRUE(parent.validate());
            for (int mutation = 0; mutation < 5; ++mutation)
            {
                SCOPED_TRACE(::testing::Message() << "type=" << static_cast<int>(type) << " mutation=" << mutation);
                auto foreign = parent;
                switch (mutation)
                {
                case 0: foreign.pp_stages[1].tp_backend = CollectiveBackendType::AUTO; break;
                case 1: foreign.pp_stages[1].tp_weights = {1.0f}; break;
                case 2: foreign.pp_stages[1].stage_devices = early.devices; break;
                case 3: foreign.pp_stages[1].has_lm_head = false; break;
                case 4: foreign.moe_rebalance.mode = MoERebalanceRuntimeMode::Observe; break;
                }
                EXPECT_THROW((void)foreign.forPipelineStage(1), std::invalid_argument);
            }
        }
    }

    /** @test A legacy dense PP plan without a child TP declaration keeps its existing AUTO policy. */
    TEST(Test__MoEOverlayPipelineStageBinding, FromPlanUnspecifiedDenseDeclarationRemainsUnspecified)
    {
        RankExecutionPlan plan;
        plan.local_pp_devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
        plan.local_pp_layer_boundaries = {0, 2, 4};
        for (const bool empty_slots : {false, true})
        {
            if (empty_slots) plan.local_pp_stage_tp_info.resize(2);
            const auto parent = Config::fromPlan(plan);
            ASSERT_TRUE(parent.validate());
            for (std::size_t index = 0; index < 2; ++index)
            {
                const auto child = parent.forPipelineStage(index);
                EXPECT_EQ(child.devices, std::vector{plan.local_pp_devices[index]});
                EXPECT_TRUE(child.weights.empty());
                EXPECT_EQ(child.backend, CollectiveBackendType::AUTO);
                EXPECT_FALSE(child.moe_routed_expert_plan);
            }
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ChildProjectionPreservesDenseRuntimeAndAuthoredTopology)
    {
        auto parent = densePipeline();
        ASSERT_TRUE(parent.validate());
        for (std::size_t index = 0; index < 2; ++index)
        {
            const auto child = parent.forPipelineStage(index);
            EXPECT_TRUE(child.validate());
            EXPECT_TRUE(child.pp_stages.empty());
            EXPECT_EQ(child.devices, parent.pp_stages[index].stage_devices);
            EXPECT_EQ(child.backend, parent.pp_stages[index].tp_backend);
            EXPECT_EQ(child.max_seq_len, 262144u);
            EXPECT_EQ(child.resident_graph_rows, 128);
            EXPECT_EQ(child.batch_size, 2);
            ASSERT_TRUE(child.nested_pp_stage_config);
            EXPECT_EQ(child.nested_pp_stage_config->first_layer, static_cast<int>(index * 2));
            EXPECT_EQ(child.nested_pp_stage_config->has_lm_head, index == 1);
            EXPECT_FALSE(child.moe_routed_expert_plan);
        }
        EXPECT_EQ(parent.pp_stages.size(), 2u);
        EXPECT_THROW((void)parent.forPipelineStage(2), std::out_of_range);
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ChildProjectionUsesIndependentStageAuthorities)
    {
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe,
                               MoERebalanceRuntimeMode::Dynamic})
        for (const int width : {1, 2, 4, 8})
        {
            StageFixture early(0, false, 0, DeviceType::ROCm, width, mode);
            StageFixture terminal(2, true, 1, DeviceType::CUDA, width, mode);
            declarePipelineGenerationDomain(early, terminal);
            auto parent = densePipeline();
            parent.moe_rebalance.mode = mode;
            parent.pp_stages = {early.stage(), terminal.stage()};
            ASSERT_TRUE(parent.validate());
            auto first = parent.forPipelineStage(0);
            auto last = parent.forPipelineStage(1);
            EXPECT_EQ(first.moe_expert_overlay_residency_authority, early.owners.authority);
            EXPECT_EQ(last.moe_expert_overlay_residency_authority, terminal.owners.authority);
            EXPECT_EQ(first.moe_expert_overlay_participant_residency, early.owners.residency);
            EXPECT_EQ(last.moe_expert_overlay_participant_residency, terminal.owners.residency);
            EXPECT_EQ(first.moe_expert_overlay_decode_histogram, early.owners.histogram);
            EXPECT_EQ(last.moe_expert_overlay_decode_histogram, terminal.owners.histogram);
            EXPECT_EQ(first.moe_routed_expert_plan->first_model_layer, 0);
            EXPECT_EQ(last.moe_routed_expert_plan->first_model_layer, 2);
            EXPECT_EQ(last.routed_expert_compute_policy, RoutedExpertComputePolicy::Apportioned);
            first.moe_routed_expert_plan->first_model_layer = 99;
            EXPECT_EQ(early.owners.authority->snapshot()->placement_plan->first_model_layer, 0);
            EXPECT_EQ(parent.forPipelineStage(0).moe_routed_expert_plan->first_model_layer, 0);
        }
    }

    /** @test Changing only the complete generation boundary cannot borrow an existing stage authority. */
    TEST(Test__MoEOverlayPipelineStageBinding, GenerationDomainIsAuthenticatedWithExactScope)
    {
        for (auto type : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (auto admitted_domain : {DeviceGenerationExecutionDomain::NativeDomain,
                                     DeviceGenerationExecutionDomain::HostedBoundary})
        {
            StageFixture fixture(2, true, 1, type, 1, MoERebalanceRuntimeMode::Dynamic);
            fixture.scope.generation_domain = admitted_domain;
            const auto binding = fixture.seal();
            EXPECT_NO_THROW(binding->requireDestination(fixture.scope, fixture.devices, {}, fixture.backend,
                MoERebalanceRuntimeMode::Dynamic));
            auto changed = fixture.scope;
            changed.generation_domain = admitted_domain == DeviceGenerationExecutionDomain::NativeDomain
                ? DeviceGenerationExecutionDomain::HostedBoundary : DeviceGenerationExecutionDomain::NativeDomain;
            EXPECT_THROW(binding->requireDestination(changed, fixture.devices, {}, fixture.backend,
                MoERebalanceRuntimeMode::Dynamic), std::invalid_argument);
            changed.generation_domain = static_cast<DeviceGenerationExecutionDomain>(255);
            EXPECT_THROW(binding->requireDestination(changed, fixture.devices, {}, fixture.backend,
                MoERebalanceRuntimeMode::Dynamic), std::invalid_argument);
        }
    }

    /** @test Ownership revalidation consumes the same complete scope used before child construction. */
    TEST(Test__MoEOverlayPipelineStageBinding, RankReauthenticationPreservesGenerationDomain)
    {
        for (auto types : {std::array{DeviceType::CUDA, DeviceType::CUDA},
                           std::array{DeviceType::ROCm, DeviceType::ROCm},
                           std::array{DeviceType::CUDA, DeviceType::ROCm},
                           std::array{DeviceType::CUDA, DeviceType::CPU}})
        {
            StageFixture first(0, false, 0, types[0], 1, MoERebalanceRuntimeMode::Dynamic);
            StageFixture last(2, true, 1, types[1], 1, MoERebalanceRuntimeMode::Dynamic,
                types[0] == types[1] ? 1 : 0);
            declarePipelineGenerationDomain(first, last);
            auto config = densePipeline();
            config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
            config.pp_stages = {first.stage(), last.stage()};
            ASSERT_TRUE(config.validate());
            const auto projected = config.forPipelineStage(0);
            ASSERT_TRUE(projected.nested_pp_stage_config);
            EXPECT_EQ(projected.nested_pp_stage_config->generation_domain, first.scope.generation_domain);
            std::vector<std::unique_ptr<IInferenceRunner>> children;
            for (const auto &device : {first.devices.front().toLocalDeviceId(), last.devices.front().toLocalDeviceId()})
                children.push_back(std::make_unique<AuthorityWitness>(device, device.is_gpu()
                    ? MoEOverlayAuthorityExecutionKind::DeviceResident : MoEOverlayAuthorityExecutionKind::HostResident));
            auto rank = RankOrchestrator::createForTestWithPipelineStages(MockModelContext::createMinimal(),
                std::move(children), config);
            EXPECT_NO_THROW(rank->requireMoEPipelineRuntimeOwnership(IInferenceRunner::MoEPipelineOwnershipValidation::OwnersBound));
            // Preserve every endpoint/owner and change only the sealed boundary.
            first.scope.generation_domain = first.scope.generation_domain == DeviceGenerationExecutionDomain::NativeDomain
                ? DeviceGenerationExecutionDomain::HostedBoundary : DeviceGenerationExecutionDomain::NativeDomain;
            auto foreign = config;
            foreign.pp_stages.front().moe_runtime = first.seal();
            EXPECT_FALSE(foreign.validate());
            EXPECT_THROW((void)foreign.forPipelineStage(0), std::invalid_argument);
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ParentRejectsModelWideExpertInheritance)
    {
        StageFixture fixture(0, false, 0, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Off);
        for (int mutation = 0; mutation < 6; ++mutation)
        {
            SCOPED_TRACE(mutation);
            auto parent = densePipeline();
            switch (mutation)
            {
            case 0: parent.moe_routed_expert_plan = std::make_shared<MoERoutedExpertPlacementPlan>(
                *fixture.owners.authority->snapshot()->placement_plan); break;
            case 1: parent.moe_expert_overlay_residency_authority = fixture.owners.authority; break;
            case 2: parent.moe_expert_overlay_participant_residency = fixture.owners.residency; break;
            case 3: parent.moe_rank_batch_transport_registry = std::make_shared<MoEOverlayRankBatchTransportRegistry>(); break;
            case 4: parent.moe_node_local_route_transport_policy.ordinary_prefill =
                        MoEOverlayNodeLocalRouteTransport::NativeCollective; break;
            case 5: parent.moe_node_local_route_transport_policy.decode =
                        MoEOverlayNodeLocalRouteTransport::NativeCollective; break;
            }
            EXPECT_FALSE(parent.validate());
        }
    }

    TEST(Test__MoEOverlayPipelineStageBinding, ParentRejectsPartialOrSwappedStageInstallation)
    {
        StageFixture early(0, false, 0, DeviceType::ROCm, 2, MoERebalanceRuntimeMode::Dynamic);
        StageFixture terminal(2, true, 1, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
        declarePipelineGenerationDomain(early, terminal);
        auto parent = densePipeline();
        parent.pp_stages = {early.stage(), terminal.stage(false)};
        EXPECT_FALSE(parent.validate());
        parent.pp_stages[1] = terminal.stage();
        std::swap(parent.pp_stages[0].moe_runtime, parent.pp_stages[1].moe_runtime);
        EXPECT_FALSE(parent.validate());
        EXPECT_THROW((void)parent.forPipelineStage(0), std::invalid_argument);
    }

    /** @test PP child projection and reuse retain each stage's exact CPU allocator/controller declaration. */
    TEST(Test__MoEOverlayPipelineStageBinding, ProjectsDistinctCpuStorageDeclarationsAcrossStageReuse)
    {
        StageFixture first(0,false,0,DeviceType::CPU,1,MoERebalanceRuntimeMode::Off,0);
        StageFixture last(2,true,0,DeviceType::CPU,1,MoERebalanceRuntimeMode::Off,1);
        declarePipelineGenerationDomain(first,last);
        const auto declaration=[](DeviceId device,int rows) {
            auto result=std::make_shared<CPUSerialExpertStorageAdmission>();
            result->allocator={0,device};
            result->participant_ids={0};
            result->controller_devices={device};
            result->envelope=std::make_shared<const CPUSerialExpertStorageEnvelope>(device,rows,1,8,2,
                std::vector<int>{1,2,4,8,16},1,8,true,CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,16,4);
            return result;
        };
        auto first_declaration=declaration(DeviceId(DeviceType::CPU,0),5);
        auto last_declaration=declaration(DeviceId(DeviceType::CPU,1),9);
        first.owners.cpu_serial_expert_storage={first_declaration};
        last.owners.cpu_serial_expert_storage={last_declaration};
        auto parent=densePipeline();
        parent.pp_stages={first.stage(),last.stage()};
        parent.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
        for(int replay=0;replay<3;++replay)
        {
            const auto retained=parent;
            const auto a=retained.forPipelineStage(0);
            const auto b=retained.forPipelineStage(1);
            ASSERT_EQ(a.cpu_serial_expert_storage.size(),1u);
            ASSERT_EQ(b.cpu_serial_expert_storage.size(),1u);
            EXPECT_TRUE(*a.cpu_serial_expert_storage.front()==*first_declaration);
            EXPECT_TRUE(*b.cpu_serial_expert_storage.front()==*last_declaration);
            EXPECT_FALSE(*a.cpu_serial_expert_storage.front()==*b.cpu_serial_expert_storage.front());
            EXPECT_NO_THROW((void)requireCPUSerialExpertStorageAdmission(a.cpu_serial_expert_storage,
                first_declaration->allocator,0,DeviceId(DeviceType::CPU,0)));
            EXPECT_THROW((void)requireCPUSerialExpertStorageAdmission(a.cpu_serial_expert_storage,
                last_declaration->allocator,0,DeviceId(DeviceType::CPU,1)),std::invalid_argument);
        }
        // Structural equality survives an independent metadata copy.
        auto copy=std::make_shared<CPUSerialExpertStorageAdmission>(*first_declaration);
        copy->envelope=std::make_shared<const CPUSerialExpertStorageEnvelope>(*first_declaration->envelope);
        EXPECT_NE(copy->envelope.get(),first_declaration->envelope.get());
        EXPECT_TRUE(*copy==*first_declaration);
        parent.cpu_serial_expert_storage={first_declaration};
        EXPECT_THROW((void)parent.forPipelineStage(0),std::invalid_argument);
    }
}


namespace llaminar2::test
{
    /** @test A two-member CPU tail cannot borrow a GPU parent's maintenance capability. */
    TEST(Test__MoEOverlayPipelineStageBinding, CpuTPTailRejectsMissingStageHostMaintenance)
    {
        StageFixture first(0, false, 0, DeviceType::CUDA, 2, MoERebalanceRuntimeMode::Dynamic);
        StageFixture last(2, true, 1, DeviceType::CPU, 2, MoERebalanceRuntimeMode::Dynamic);
        declarePipelineGenerationDomain(first, last);
        auto config = densePipeline();
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        config.pp_stages = {first.stage(), last.stage()};
        MockLocalTPContext::Config context_config;
        context_config.devices = first.devices; context_config.backend = CollectiveBackendType::NCCL;
        auto first_context = std::make_unique<MockLocalTPContext>(context_config);
        std::vector<std::unique_ptr<IInferenceRunner>> gpu;
        gpu.push_back(std::make_unique<AuthorityWitness>(DeviceId::cuda(0), MoEOverlayAuthorityExecutionKind::DeviceResident));
        gpu.push_back(std::make_unique<AuthorityWitness>(DeviceId::cuda(1), MoEOverlayAuthorityExecutionKind::DeviceResident));
        auto first_child = RankOrchestrator::createForTest(MockModelContext::createMinimal(), std::move(gpu),
            std::move(first_context), config.forPipelineStage(0));
        context_config.devices = last.devices; context_config.backend = CollectiveBackendType::HOST;
        auto last_context = std::make_unique<MockLocalTPContext>(context_config);
        std::vector<std::unique_ptr<IInferenceRunner>> cpu;
        // Execution DeviceId names the host backend. Exact NUMA membership stays
        // in the authored context; inventing CPU:1 would reject before ownership.
        for (const auto &participant : last.devices)
            cpu.push_back(std::make_unique<AuthorityWitness>(participant.toLocalDeviceId(),
                MoEOverlayAuthorityExecutionKind::HostResident));
        auto last_child = RankOrchestrator::createForTest(MockModelContext::createMinimal(), std::move(cpu),
            std::move(last_context), config.forPipelineStage(1));
        std::vector<std::unique_ptr<IInferenceRunner>> children;
        children.push_back(std::move(first_child)); children.push_back(std::move(last_child));
        auto rank = RankOrchestrator::createForTestWithPipelineStages(MockModelContext::createMinimal(), std::move(children), config);
        try
        {
            rank->requireMoEPipelineRuntimeOwnership(IInferenceRunner::MoEPipelineOwnershipValidation::OwnersBound);
            FAIL() << "A movable CPU TP tail was admitted without its stage maintenance owner";
        }
        catch (const std::logic_error &error)
        {
            EXPECT_STREQ(error.what(), "Movable host MoE pipeline stage has no admitted maintenance owner");
        }
    }
}
