/**
 * @file Test__MoEOverlayParticipantResidency.cpp
 * @brief Device-free RCU tests for participant-local prepared expert banks.
 */

#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayParticipantMigration.h"
#include "execution/moe/DecodeExpertHistogram.h"

#include "loaders/ExpertGemmRegistry.h"
#include "tensors/TensorKernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Identity-only GEMM lifetime used by the bank protocol tests. */
    class IdentityGemm final : public ITensorGemm
    {
    public:
        explicit IdentityGemm(int identity)
            : identity_(identity)
        {
        }

        bool supports_device(int) const override { return true; }

        bool multiply_tensor(
            const TensorBase *,
            TensorBase *,
            int,
            int,
            int,
            bool,
            float,
            float,
            const TensorBase *,
            const IMPIContext *,
            int,
            DeviceWorkspaceManager *,
            int) override
        {
            return false;
        }

        int identity() const noexcept { return identity_; }

    private:
        int identity_ = 0;
    };

    MoEOverlayPreparedExpertTriplet triplet(int base)
    {
        return {
            .gate = std::make_shared<IdentityGemm>(base + 1),
            .up = std::make_shared<IdentityGemm>(base + 2),
            .down = std::make_shared<IdentityGemm>(base + 3),
        };
    }

    MoEOverlayParticipantResidencyBank initialBank(
        const MoEOverlayPreparedExpertTriplet &expert_zero)
    {
        MoEOverlayParticipantResidencyBank bank;
        bank.epoch = 1;
        bank.participant_id = 4;
        bank.device = DeviceId::cpu();
        bank.layers.resize(1);
        bank.layers[0].resident_mask.assign(2, false);
        bank.layers[0].experts.resize(2);
        bank.layers[0].setResidentExpert(0, expert_zero);
        return bank;
    }

    MoEExpertOwnerMap twoParticipantOwnerMap()
    {
        RoutedExpertDomain first_domain;
        first_domain.name = "first";
        first_domain.scope = ExecutionDomainScope::SINGLE;
        first_domain.backend = CollectiveBackendType::HOST;
        first_domain.participants = {GlobalDeviceAddress::cpu(0)};
        first_domain.world_ranks = {0};
        first_domain.owner_rank = 0;

        RoutedExpertDomain second_domain;
        second_domain.name = "second";
        second_domain.scope = ExecutionDomainScope::SINGLE;
        second_domain.backend = CollectiveBackendType::HOST;
        second_domain.participants = {GlobalDeviceAddress::cpu(1)};
        second_domain.world_ranks = {1};
        second_domain.owner_rank = 1;

        RoutedExpertTier first_tier;
        first_tier.name = "hot";
        first_tier.domain = "first";
        first_tier.priority = 0;
        first_tier.max_experts_per_layer = 1;

        RoutedExpertTier second_tier;
        second_tier.name = "cold";
        second_tier.domain = "second";
        second_tier.priority = 1;
        second_tier.fallback = true;

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "first";
        plan.shared_expert_domain = "first";
        plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan.domains = {std::move(first_domain), std::move(second_domain)};
        plan.routed_tiers = {std::move(first_tier), std::move(second_tier)};
        plan.placements = {{
            .layer = 0,
            .routed_expert_tier = {0, 1},
        }};
        return MoEExpertOwnerMap::build(plan);
    }

    /** @brief Construct a dynamic two-tier plan with one expert in each tier. */
    MoERoutedExpertPlacementPlan dynamicTwoParticipantPlan()
    {
        RoutedExpertDomain hot;
        hot.name = "hot_domain";
        hot.scope = ExecutionDomainScope::SINGLE;
        hot.backend = CollectiveBackendType::HOST;
        hot.participants = {GlobalDeviceAddress::cpu(0)};
        hot.world_ranks = {0};
        hot.owner_rank = 0;
        hot.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;

        RoutedExpertDomain cold;
        cold.name = "cold_domain";
        cold.scope = ExecutionDomainScope::SINGLE;
        cold.backend = CollectiveBackendType::HOST;
        cold.participants = {GlobalDeviceAddress::cpu(1)};
        cold.world_ranks = {0};
        cold.owner_rank = 0;
        cold.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;

        RoutedExpertTier hot_tier;
        hot_tier.name = "hot";
        hot_tier.domain = "hot_domain";
        hot_tier.priority = 0;
        hot_tier.max_experts_per_layer = 1;

        RoutedExpertTier cold_tier;
        cold_tier.name = "cold";
        cold_tier.domain = "cold_domain";
        cold_tier.priority = 1;
        cold_tier.fallback = true;

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "hot_domain";
        plan.shared_expert_domain = "hot_domain";
        plan.residency_policy =
            RoutedExpertResidencyPolicy::RoutedTierRebalanced;
        plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
        plan.domains = {std::move(hot), std::move(cold)};
        plan.routed_tiers = {std::move(hot_tier), std::move(cold_tier)};
        plan.placements = {{
            .layer = 0,
            .routed_expert_tier = {0, 1},
        }};
        return plan;
    }

    /** @brief Ready device-free transfer retaining its prepared engine lifetime. */
    class ReadyProjectionTransfer final
        : public IMoEOverlayTierTransferOperation
    {
    public:
        MoEOverlayResidencyWaveProgress poll(std::string *) noexcept override
        {
            return aborted_ ? MoEOverlayResidencyWaveProgress::Failed
                            : MoEOverlayResidencyWaveProgress::Ready;
        }

        void abort() noexcept override { aborted_ = true; }

    private:
        bool aborted_ = false;
    };

    /** @brief Scripted physical provider that publishes complete or partial arrivals. */
    class IdentityTransferProvider final
        : public IMoEOverlayParticipantTransferProvider
    {
    public:
        explicit IdentityTransferProvider(bool omit_down = false)
            : omit_down_(omit_down)
        {
        }

        MoEOverlayParticipantPreparedTransfers prepareTransfers(
            const MoEOverlayResidencyTransaction &transaction,
            const std::vector<int> &local_destination_participants) override
        {
            ++prepare_calls;
            MoEOverlayParticipantPreparedTransfers prepared;
            prepared.status = MoEOverlayResidencyStageStartStatus::Started;
            prepared.migrations.resize(transaction.migrations.size());
            for (std::size_t index = 0;
                 index < transaction.migrations.size();
                 ++index)
            {
                auto &entry = prepared.migrations[index];
                const auto &migration = transaction.migrations[index];
                if (!std::binary_search(
                        local_destination_participants.begin(),
                        local_destination_participants.end(),
                        migration.destination.owner_participant))
                {
                    for (auto &projection : entry.projections)
                    {
                        projection =
                            std::make_unique<ReadyProjectionTransfer>();
                    }
                    continue;
                }

                entry.destination_arrival =
                    std::make_shared<MoEOverlayPreparedExpertArrival>();
                const auto engines = triplet(
                    2000 + migration.destination.owner_participant * 100 +
                    migration.expert_id * 10);
                entry.projections[0] =
                    std::make_unique<MoEOverlayPreparedProjectionOperation>(
                        std::make_unique<ReadyProjectionTransfer>(),
                        entry.destination_arrival,
                        ExpertTierWeightProjection::Gate,
                        engines.gate);
                entry.projections[1] =
                    std::make_unique<MoEOverlayPreparedProjectionOperation>(
                        std::make_unique<ReadyProjectionTransfer>(),
                        entry.destination_arrival,
                        ExpertTierWeightProjection::Up,
                        engines.up);
                entry.projections[2] = omit_down_
                                           ? std::unique_ptr<
                                                 IMoEOverlayTierTransferOperation>(
                                                 std::make_unique<
                                                     ReadyProjectionTransfer>())
                                           : std::unique_ptr<
                                                 IMoEOverlayTierTransferOperation>(
                                                 std::make_unique<
                                                     MoEOverlayPreparedProjectionOperation>(
                                                     std::make_unique<
                                                         ReadyProjectionTransfer>(),
                                                     entry.destination_arrival,
                                                     ExpertTierWeightProjection::Down,
                                                     engines.down));
            }
            return prepared;
        }

        /** @brief Identity fixtures own no physical slot arena to recycle. */
        void retirePreviousSources(
            std::uint64_t,
            const std::vector<MoEOverlayTierMigration> &) noexcept override
        {
        }

        int prepare_calls = 0;

    private:
        bool omit_down_ = false;
    };

    /** @brief Authority, histogram, banks, and expected swap transaction. */
    struct ParticipantMigrationFixture
    {
        std::shared_ptr<DecodeExpertHistogram> histogram;
        std::shared_ptr<MoEOverlayResidencyAuthority> authority;
        std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
        MoEOverlayResidencyTransaction transaction;
    };

    /** @brief Create initial p0:e0 / p1:e1 banks and a hotter-e1 swap proposal. */
    ParticipantMigrationFixture participantMigrationFixture()
    {
        ParticipantMigrationFixture fixture;
        const auto plan = dynamicTwoParticipantPlan();
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = 2;
        histogram_config.top_k = 1;
        histogram_config.window_size = 2;
        histogram_config.sockets = {DeviceId::cpu(), DeviceId::cpu()};
        histogram_config.ownership = owner_map.layeredOwnership(1, 2);
        fixture.histogram =
            std::make_shared<DecodeExpertHistogram>(histogram_config);
        const std::uint64_t counts[] = {1, 100};
        fixture.histogram->mergeLayerCounts(0, counts, 2, false);

        fixture.authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = plan,
                .model_metadata = {
                    .num_layers = 1,
                    .num_experts = 2,
                    .d_model = 64,
                    .routed_intermediate_size = 32,
                    .routed_quant_type = "Q4_0",
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = fixture.histogram.get(),
                .perf_device = "cpu-hot/cpu-cold",
            });
        fixture.registry =
            std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = owner_map,
                    .local_participant_ids = {0, 1},
                    .num_layers = 1,
                    .num_experts = 2,
                    .initial_epoch = 1,
                    .retained_epoch_capacity = 2,
                });

        std::string error;
        for (const int participant_id : std::vector<int>{0, 1})
        {
            const auto mask = owner_map.expertMaskForParticipant(
                0, participant_id, 2);
            std::vector<MoEOverlayPreparedExpertTriplet> engines(2);
            for (int expert = 0; expert < 2; ++expert)
            {
                if (mask[static_cast<std::size_t>(expert)])
                {
                    engines[static_cast<std::size_t>(expert)] =
                        triplet(1700 + participant_id * 100 + expert * 10);
                }
            }
            if (!fixture.registry->registerInitialLayer(
                    participant_id, 0, mask, engines, &error))
            {
                throw std::runtime_error(error);
            }
        }
        fixture.transaction = fixture.authority->proposeFromHistogram();
        if (!fixture.transaction.valid() ||
            fixture.transaction.migrations.size() != 2)
        {
            throw std::runtime_error(
                "Participant migration fixture did not produce a two-edge swap");
        }
        return fixture;
    }
} // namespace

TEST(Test__MoEOverlayParticipantResidency,
     RetainsOldAndCandidateBanksByExactEpoch)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });

    const auto old_expert = triplet(100);
    auto epoch_one = initialBank(old_expert);
    std::string error;
    EXPECT_EQ(
        residency.installReadyBank(epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    auto epoch_two = residency.cloneCandidate(1, 2);
    epoch_two.layers[0].clearExpert(0);
    const auto new_expert = triplet(200);
    epoch_two.layers[0].setResidentExpert(1, new_expert);
    EXPECT_EQ(
        residency.installReadyBank(epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    const auto acquired_one = residency.acquire(1);
    const auto acquired_two = residency.acquire(2);
    ASSERT_NE(acquired_one, nullptr);
    ASSERT_NE(acquired_two, nullptr);
    EXPECT_TRUE(acquired_one->layers[0].resident_mask[0]);
    EXPECT_FALSE(acquired_one->layers[0].resident_mask[1]);
    EXPECT_FALSE(acquired_two->layers[0].resident_mask[0]);
    EXPECT_TRUE(acquired_two->layers[0].resident_mask[1]);
    EXPECT_EQ(
        acquired_one->layers[0].experts[0].gate.get(),
        old_expert.gate.get());
    EXPECT_EQ(
        acquired_two->layers[0].experts[1].gate.get(),
        new_expert.gate.get());
}

TEST(Test__MoEOverlayParticipantResidency,
     BoundedTwoBankCapacityDefersUntilOldEpochRetires)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });
    std::string error;
    auto epoch_one = initialBank(triplet(300));
    ASSERT_EQ(
        residency.installReadyBank(epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    auto epoch_two = residency.cloneCandidate(1, 2);
    ASSERT_EQ(
        residency.installReadyBank(epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);

    auto epoch_three = residency.cloneCandidate(2, 3);
    EXPECT_FALSE(residency.hasCandidateCapacity());
    EXPECT_EQ(
        residency.installReadyBank(epoch_three, &error),
        MoEOverlayParticipantBankInstallStatus::CapacityUnavailable);
    EXPECT_EQ(residency.retainedEpochCount(), 2u);

    EXPECT_TRUE(residency.retire(1));
    EXPECT_TRUE(residency.hasCandidateCapacity());
    EXPECT_EQ(
        residency.installReadyBank(epoch_three, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;
    EXPECT_EQ(residency.acquire(1), nullptr);
    EXPECT_NE(residency.acquire(2), nullptr);
    EXPECT_NE(residency.acquire(3), nullptr);
}

TEST(Test__MoEOverlayParticipantResidency,
     OldEngineLifetimeSurvivesPublicationUntilRetirement)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });
    std::string error;
    auto old_expert = triplet(400);
    std::weak_ptr<ITensorGemm> old_gate = old_expert.gate;
    auto epoch_one = initialBank(old_expert);
    ASSERT_EQ(
        residency.installReadyBank(epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    auto epoch_two = residency.cloneCandidate(1, 2);
    epoch_two.layers[0].clearExpert(0);
    epoch_two.layers[0].setResidentExpert(1, triplet(500));
    ASSERT_EQ(
        residency.installReadyBank(epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);

    /* Drop every construction-time copy; epoch one is now the only owner. */
    old_expert = {};
    epoch_one = {};
    EXPECT_FALSE(old_gate.expired());

    EXPECT_TRUE(residency.retire(1));
    EXPECT_TRUE(old_gate.expired());
}

TEST(Test__MoEOverlayParticipantResidency,
     InstallationIsIdempotentButRejectsSameEpochWithDifferentIdentity)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });
    std::string error;
    const auto epoch_one = initialBank(triplet(600));
    ASSERT_EQ(
        residency.installReadyBank(epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    EXPECT_EQ(
        residency.installReadyBank(epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::AlreadyInstalled);

    auto conflicting = epoch_one;
    conflicting.layers[0].clearExpert(0);
    conflicting.layers[0].setResidentExpert(1, triplet(700));
    EXPECT_EQ(
        residency.installReadyBank(conflicting, &error),
        MoEOverlayParticipantBankInstallStatus::EpochConflict);
    EXPECT_FALSE(error.empty());
}

TEST(Test__MoEOverlayParticipantResidency,
     InvalidPartialResidentBankFailsClosed)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });
    auto invalid = initialBank(triplet(800));
    invalid.layers[0].resident_mask[1] = true;
    invalid.layers[0].experts[1].gate =
        std::make_shared<IdentityGemm>(900);

    std::string error;
    EXPECT_EQ(
        residency.installReadyBank(invalid, &error),
        MoEOverlayParticipantBankInstallStatus::Invalid);
    EXPECT_EQ(residency.retainedEpochCount(), 0u);
    EXPECT_FALSE(error.empty());
}

TEST(Test__MoEOverlayParticipantResidency,
     AbortRemovesOnlyNamedUnpublishedCandidate)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });
    std::string error;
    const auto epoch_one = initialBank(triplet(1000));
    ASSERT_EQ(
        residency.installReadyBank(epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    const auto epoch_two = residency.cloneCandidate(1, 2);
    ASSERT_EQ(
        residency.installReadyBank(epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);

    EXPECT_TRUE(residency.abortUnpublished(2));
    EXPECT_FALSE(residency.abortUnpublished(2));
    EXPECT_NE(residency.acquire(1), nullptr);
    EXPECT_EQ(residency.acquire(2), nullptr);
}

TEST(Test__MoEOverlayParticipantResidencyRegistry,
     AssemblesCanonicalInitialBanksLayerByLayer)
{
    MoEOverlayParticipantResidencyRegistry registry({
        .owner_map = twoParticipantOwnerMap(),
        .local_participant_ids = {1, 0},
        .num_layers = 1,
        .num_experts = 2,
        .initial_epoch = 1,
    });
    EXPECT_EQ(registry.localParticipantIds(), (std::vector<int>{0, 1}));
    EXPECT_FALSE(registry.allInitialBanksReady());
    {
        const auto deficits = registry.incompleteInitialBanks();
        ASSERT_EQ(deficits.size(), 2u);
        EXPECT_EQ(deficits[0].participant_id, 0);
        EXPECT_EQ(deficits[0].device, DeviceId::cpu());
        EXPECT_EQ(deficits[0].missing_layers, (std::vector<int>{0}));
        EXPECT_FALSE(deficits[0].publication_pending);
        EXPECT_EQ(deficits[1].participant_id, 1);
        EXPECT_EQ(deficits[1].missing_layers, (std::vector<int>{0}));
    }

    std::vector<MoEOverlayPreparedExpertTriplet> first_engines(2);
    first_engines[0] = triplet(1100);
    std::string error;
    EXPECT_TRUE(registry.registerInitialLayer(
        0,
        0,
        {true, false},
        first_engines,
        &error))
        << error;
    EXPECT_FALSE(registry.allInitialBanksReady());
    {
        const auto deficits = registry.incompleteInitialBanks();
        ASSERT_EQ(deficits.size(), 1u);
        EXPECT_EQ(deficits[0].participant_id, 1);
        EXPECT_EQ(deficits[0].missing_layers, (std::vector<int>{0}));
    }

    std::vector<MoEOverlayPreparedExpertTriplet> second_engines(2);
    second_engines[1] = triplet(1200);
    EXPECT_TRUE(registry.registerInitialLayer(
        1,
        0,
        {false, true},
        second_engines,
        &error))
        << error;
    EXPECT_TRUE(registry.allInitialBanksReady());
    EXPECT_TRUE(registry.incompleteInitialBanks().empty());

    const auto first = registry.endpoint(0);
    const auto second = registry.endpoint(1);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_TRUE(first->acquire(1)->layers[0].resident_mask[0]);
    EXPECT_TRUE(second->acquire(1)->layers[0].resident_mask[1]);

    /* Identical graph-cache registration is idempotent. */
    EXPECT_TRUE(registry.registerInitialLayer(
        0,
        0,
        {true, false},
        first_engines,
        &error))
        << error;
}

TEST(Test__MoEOverlayParticipantResidencyRegistry,
     RejectsTacticalMaskOrEngineIdentityDrift)
{
    MoEOverlayParticipantResidencyRegistry registry({
        .owner_map = twoParticipantOwnerMap(),
        .local_participant_ids = {0},
        .num_layers = 1,
        .num_experts = 2,
        .initial_epoch = 1,
    });
    std::vector<MoEOverlayPreparedExpertTriplet> engines(2);
    engines[0] = triplet(1300);
    std::string error;
    EXPECT_FALSE(registry.registerInitialLayer(
        0,
        0,
        {false, true},
        engines,
        &error));
    EXPECT_FALSE(error.empty());

    ASSERT_TRUE(registry.registerInitialLayer(
        0,
        0,
        {true, false},
        engines,
        &error))
        << error;
    auto different = engines;
    different[0] = triplet(1400);
    EXPECT_FALSE(registry.registerInitialLayer(
        0,
        0,
        {true, false},
        different,
        &error));
    EXPECT_FALSE(error.empty());
}

TEST(Test__MoEOverlayParticipantResidencyRegistry,
     RelayOnlyRankOwnsAValidVacuouslyReadyRegistry)
{
    MoEOverlayParticipantResidencyRegistry registry({
        .owner_map = twoParticipantOwnerMap(),
        .local_participant_ids = {},
        .num_layers = 1,
        .num_experts = 2,
        .initial_epoch = 1,
    });

    EXPECT_TRUE(registry.localParticipantIds().empty());
    EXPECT_TRUE(registry.allInitialBanksReady());
    EXPECT_TRUE(registry.incompleteInitialBanks().empty());
    EXPECT_EQ(registry.endpoint(0), nullptr);
    EXPECT_EQ(registry.endpoint(1), nullptr);
}

TEST(Test__MoEOverlayParticipantResidencyRegistry,
     ResolvesExactOwnedParticipantTripletsAndRejectsPartialRoles)
{
    const auto owner_map = twoParticipantOwnerMap();
    const auto *participant = owner_map.participantForId(0);
    ASSERT_NE(participant, nullptr);

    ExpertGemmRegistry engines;
    const auto gate = std::make_shared<IdentityGemm>(1501);
    const auto up = std::make_shared<IdentityGemm>(1502);
    const auto down = std::make_shared<IdentityGemm>(1503);
    const int world_rank = participant->world_rank_known
                               ? participant->world_rank
                               : -1;
    const auto register_role = [&](ExpertGemmRegistry::WeightRole role,
                                   const std::shared_ptr<IdentityGemm> &engine)
    {
        engines.registerEngineForParticipant(
            participant->domain_name,
            participant->device,
            world_rank,
            participant->domain_participant_index,
            0,
            0,
            role,
            engine.get(),
            engine);
    };
    register_role(ExpertGemmRegistry::WeightRole::GATE, gate);
    register_role(ExpertGemmRegistry::WeightRole::UP, up);
    register_role(ExpertGemmRegistry::WeightRole::DOWN, down);

    std::vector<MoEOverlayPreparedExpertTriplet> resolved;
    std::string error;
    ASSERT_TRUE(resolveMoEOverlayPreparedExpertTriplets(
        engines,
        *participant,
        0,
        2,
        {true, false},
        resolved,
        &error))
        << error;
    ASSERT_EQ(resolved.size(), 2u);
    EXPECT_EQ(resolved[0].gate.get(), gate.get());
    EXPECT_EQ(resolved[0].up.get(), up.get());
    EXPECT_EQ(resolved[0].down.get(), down.get());
    EXPECT_TRUE(resolved[1].empty());

    EXPECT_FALSE(resolveMoEOverlayPreparedExpertTriplets(
        engines,
        *participant,
        0,
        2,
        {true, true},
        resolved,
        &error));
    EXPECT_TRUE(resolved.empty());
    EXPECT_FALSE(error.empty());
}

TEST(Test__MoEOverlayPreparedExpertArrival,
     PublishesOneExactCompleteTripletAndRejectsIdentityDrift)
{
    MoEOverlayPreparedExpertArrival arrival;
    const auto engines = triplet(3000);
    std::string error;
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Gate, engines.gate, &error))
        << error;
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Gate, engines.gate, &error))
        << error;
    EXPECT_FALSE(arrival.publish(
        ExpertTierWeightProjection::Gate, triplet(3100).gate, &error));
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Up, engines.up, &error))
        << error;

    MoEOverlayPreparedExpertTriplet resolved;
    EXPECT_FALSE(arrival.completeTriplet(resolved, &error));
    EXPECT_TRUE(resolved.empty());
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Down, engines.down, &error))
        << error;
    ASSERT_TRUE(arrival.completeTriplet(resolved, &error)) << error;
    EXPECT_TRUE(resolved.sameIdentity(engines));

    EXPECT_TRUE(arrival.fail("late physical failure"));
    EXPECT_FALSE(arrival.completeTriplet(resolved, &error));
    EXPECT_EQ(error, "late physical failure");
}

TEST(Test__MoEOverlayParticipantMigration,
     PublishesCompleteSwappedBanksWhileOldTicketRetainsOldEpoch)
{
    auto fixture = participantMigrationFixture();
    auto old_ticket = fixture.authority->tryAcquireTicketSnapshot();
    ASSERT_TRUE(old_ticket.has_value());
    ASSERT_EQ((*old_ticket)->epoch, 1u);

    auto provider = std::make_shared<IdentityTransferProvider>();
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = fixture.registry,
        .transfer_provider = provider,
        .perf_device = "cpu-hot/cpu-cold",
    });
    MoEOverlayTierMigrationTransport transport({
        .factory = &factory,
        .projections_per_expert = kMoEOverlayExpertProjectionCount,
        .perf_device = "cpu-hot/cpu-cold",
    });

    const auto started = fixture.authority->beginApply(
        fixture.transaction, transport);
    ASSERT_EQ(started.status, MoEOverlayResidencyApplyStatus::Started)
        << started.error;
    EXPECT_EQ(provider->prepare_calls, 1);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

    const auto committing = fixture.authority->advanceBackground();
    ASSERT_EQ(
        committing.status, MoEOverlayResidencyApplyStatus::Committing)
        << committing.error;
    /* Candidate banks are complete before, but never visible to, epoch one. */
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(2), nullptr);
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(2), nullptr);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

    const auto committed = fixture.authority->advanceBackground();
    ASSERT_EQ(committed.status, MoEOverlayResidencyApplyStatus::Committed)
        << committed.error;
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 2u);
    EXPECT_EQ(fixture.authority->pendingRetirementCount(), 1u);

    const auto p0_old = fixture.registry->endpoint(0)->acquire(1);
    const auto p0_new = fixture.registry->endpoint(0)->acquire(2);
    const auto p1_old = fixture.registry->endpoint(1)->acquire(1);
    const auto p1_new = fixture.registry->endpoint(1)->acquire(2);
    ASSERT_NE(p0_old, nullptr);
    ASSERT_NE(p0_new, nullptr);
    ASSERT_NE(p1_old, nullptr);
    ASSERT_NE(p1_new, nullptr);
    EXPECT_EQ(p0_old->layers[0].resident_mask,
              (std::vector<bool>{true, false}));
    EXPECT_EQ(p0_new->layers[0].resident_mask,
              (std::vector<bool>{false, true}));
    EXPECT_EQ(p1_old->layers[0].resident_mask,
              (std::vector<bool>{false, true}));
    EXPECT_EQ(p1_new->layers[0].resident_mask,
              (std::vector<bool>{true, false}));

    /* A live old sparse ticket is the sole retirement barrier. */
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(1), nullptr);
    old_ticket.reset();
    EXPECT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Idle);
    EXPECT_EQ(fixture.registry->endpoint(0)->acquire(1), nullptr);
    EXPECT_EQ(fixture.registry->endpoint(1)->acquire(1), nullptr);
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(2), nullptr);

    const auto authority_stats = fixture.authority->stats();
    EXPECT_EQ(authority_stats.committed_migrations, 2u);
    EXPECT_EQ(authority_stats.promotions, 1u);
    EXPECT_EQ(authority_stats.demotions, 1u);
    const auto transport_stats = transport.stats();
    EXPECT_EQ(transport_stats.transfer_operations_completed, 6u);
    EXPECT_EQ(transport_stats.inference_stream_waits, 0u);
    EXPECT_EQ(transport_stats.blocking_synchronizations, 0u);
}

TEST(Test__MoEOverlayParticipantMigration,
     IncompleteArrivalFailsCommitAndNeverExposesCandidateEpoch)
{
    auto fixture = participantMigrationFixture();
    auto provider = std::make_shared<IdentityTransferProvider>(
        /*omit_down=*/true);
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = fixture.registry,
        .transfer_provider = provider,
        .perf_device = "cpu-hot/cpu-cold",
    });
    MoEOverlayTierMigrationTransport transport({
        .factory = &factory,
        .projections_per_expert = kMoEOverlayExpertProjectionCount,
        .perf_device = "cpu-hot/cpu-cold",
    });

    ASSERT_EQ(
        fixture.authority
            ->beginApply(fixture.transaction, transport)
            .status,
        MoEOverlayResidencyApplyStatus::Started);
    const auto failed = fixture.authority->advanceBackground();
    EXPECT_EQ(failed.status, MoEOverlayResidencyApplyStatus::CommitFailed);
    EXPECT_NE(failed.error.find("gate/up/down"), std::string::npos);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
    EXPECT_EQ(fixture.registry->endpoint(0)->acquire(2), nullptr);
    EXPECT_EQ(fixture.registry->endpoint(1)->acquire(2), nullptr);
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(1), nullptr);
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(1), nullptr);
    EXPECT_EQ(transport.stats().waves_aborted, 1u);
}

TEST(Test__MoEOverlayParticipantMigration,
     CandidateCapacityDefersWholeCycleBeforePhysicalProviderRuns)
{
    auto fixture = participantMigrationFixture();
    const auto endpoint = fixture.registry->endpoint(0);
    ASSERT_NE(endpoint, nullptr);
    auto unrelated_candidate = endpoint->cloneCandidate(1, 99);
    std::string error;
    ASSERT_EQ(
        endpoint->installReadyBank(unrelated_candidate, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    auto provider = std::make_shared<IdentityTransferProvider>();
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = fixture.registry,
        .transfer_provider = provider,
        .perf_device = "cpu-hot/cpu-cold",
    });
    const auto prepared = factory.prepare(fixture.transaction);
    EXPECT_EQ(
        prepared.status, MoEOverlayResidencyStageStartStatus::Deferred);
    EXPECT_TRUE(prepared.transfers.empty());
    EXPECT_EQ(prepared.inactive_bank, nullptr);
    EXPECT_EQ(provider->prepare_calls, 0);
    EXPECT_NE(prepared.error.find("inactive RCU bank"), std::string::npos);
}

TEST(Test__MoEOverlayParticipantMigration,
     CompetingEpochInstallationRollsBackOnlyThisTransactionsPrefix)
{
    auto fixture = participantMigrationFixture();
    auto provider = std::make_shared<IdentityTransferProvider>();
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = fixture.registry,
        .transfer_provider = provider,
        .perf_device = "cpu-hot/cpu-cold",
    });
    auto prepared = factory.prepare(fixture.transaction);
    ASSERT_EQ(
        prepared.status, MoEOverlayResidencyStageStartStatus::Started)
        << prepared.error;
    ASSERT_NE(prepared.inactive_bank, nullptr);

    /*
     * Race a separate authority onto p1 after preflight. p0 is installed first
     * by the tested transaction; the p1 AlreadyInstalled result must roll p0
     * back without deleting the independently owned p1 epoch.
     */
    const auto p1 = fixture.registry->endpoint(1);
    ASSERT_NE(p1, nullptr);
    auto external = p1->cloneCandidate(1, 2);
    external.layers[0].clearExpert(1);
    external.layers[0].setResidentExpert(0, triplet(4000));
    std::string error;
    ASSERT_EQ(
        p1->installReadyBank(external, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    for (auto &operation : prepared.transfers)
    {
        ASSERT_NE(operation, nullptr);
        EXPECT_EQ(
            operation->poll(&error),
            MoEOverlayResidencyWaveProgress::Ready)
            << error;
    }
    EXPECT_FALSE(prepared.inactive_bank->beginCommit(&error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(fixture.registry->endpoint(0)->acquire(2), nullptr);
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(2), nullptr);
    EXPECT_EQ(fixture.registry->endpoint(0)->retainedEpochCount(), 1u);
    EXPECT_EQ(fixture.registry->endpoint(1)->retainedEpochCount(), 2u);

    prepared.inactive_bank->abort();
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(2), nullptr);
}

TEST(Test__MoEOverlayParticipantMigration,
     RelayOnlyRankParticipatesWithoutInventingRemotePreparedEngines)
{
    auto fixture = participantMigrationFixture();
    auto relay_registry =
        std::make_shared<MoEOverlayParticipantResidencyRegistry>(
            MoEOverlayParticipantResidencyRegistry::Config{
                .owner_map = fixture.transaction.previous->owner_map,
                .local_participant_ids = {},
                .num_layers = 1,
                .num_experts = 2,
                .initial_epoch = 1,
            });
    auto provider = std::make_shared<IdentityTransferProvider>();
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = relay_registry,
        .transfer_provider = provider,
        .perf_device = "relay",
    });
    auto prepared = factory.prepare(fixture.transaction);
    ASSERT_EQ(
        prepared.status, MoEOverlayResidencyStageStartStatus::Started)
        << prepared.error;
    ASSERT_EQ(prepared.transfers.size(), 6u);
    for (auto &operation : prepared.transfers)
    {
        EXPECT_EQ(
            operation->poll(nullptr),
            MoEOverlayResidencyWaveProgress::Ready);
    }
    ASSERT_TRUE(prepared.inactive_bank->beginCommit(nullptr));
    EXPECT_EQ(
        prepared.inactive_bank->pollCommit(nullptr),
        MoEOverlayResidencyWaveProgress::Ready);
    prepared.inactive_bank->retirePrevious();
    EXPECT_TRUE(relay_registry->localParticipantIds().empty());
}

TEST(Test__MoEOverlayParticipantResidency,
     ServiceMeasurementsAreCoherentTryOwnedAndPhaseSeparated)
{
    MoEOverlayParticipantResidency endpoint({
        .participant_id = 7,
        .device = DeviceId::cpu(),
        .num_layers = 2,
        .num_experts = 4,
        .collect_economy_service_measurements = true,
    });
    ASSERT_TRUE(endpoint.collectsEconomyServiceMeasurements());
    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::DecodeToken, 101, 2),
        MoEOverlayServiceMeasurementRecordStatus::Recorded);
    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::DecodeToken, 202, 3),
        MoEOverlayServiceMeasurementRecordStatus::Recorded);
    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::PrefillChunk, 303, 4),
        MoEOverlayServiceMeasurementRecordStatus::Recorded);
    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::GroupedVerifier, 404, 5),
        MoEOverlayServiceMeasurementRecordStatus::Recorded);

    std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
    ASSERT_TRUE(endpoint.trySnapshotServiceMeasurements(&rows));
    ASSERT_EQ(rows.size(), 2u);
    EXPECT_TRUE(rows[0].valid());
    EXPECT_EQ(rows[0].participant_id, 7);
    EXPECT_EQ(rows[0].layer, 0);
    EXPECT_EQ(rows[0].total_nanoseconds[0], 303u);
    EXPECT_EQ(rows[0].activation_count[0], 5u);
    EXPECT_EQ(rows[0].sample_count[0], 2u);
    EXPECT_EQ(rows[0].total_nanoseconds[1], 303u);
    EXPECT_EQ(rows[0].activation_count[1], 4u);
    EXPECT_EQ(rows[0].sample_count[1], 1u);
    EXPECT_EQ(rows[0].total_nanoseconds[2], 404u);
    EXPECT_EQ(rows[0].activation_count[2], 5u);
    EXPECT_EQ(rows[0].sample_count[2], 1u);
    EXPECT_TRUE(rows[1].valid());
    EXPECT_EQ(rows[1].sample_count,
              (std::array<uint64_t,
                          kExpertHistogramProductionSourceCount>{}));

    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::SyntheticTest, 1, 1),
        MoEOverlayServiceMeasurementRecordStatus::Invalid);
    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            2, ExpertHistogramSource::DecodeToken, 1, 1),
        MoEOverlayServiceMeasurementRecordStatus::Invalid);
}

TEST(Test__MoEOverlayParticipantResidency,
     ServiceMeasurementContentionDropsInsteadOfWaitingAndOverflowIsFatal)
{
    MoEOverlayParticipantResidency endpoint({
        .participant_id = 0,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 1,
        .collect_economy_service_measurements = true,
    });
    constexpr int kThreads = 8;
    constexpr int kCallsPerThread = 2000;
    std::atomic<bool> start{false};
    std::atomic<uint64_t> recorded{0};
    std::atomic<uint64_t> contended{0};
    std::atomic<uint64_t> unexpected{0};
    std::vector<std::thread> workers;
    for (int thread = 0; thread < kThreads; ++thread)
    {
        workers.emplace_back(
            [&]
            {
                while (!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                for (int call = 0; call < kCallsPerThread; ++call)
                {
                    const auto status = endpoint.recordServiceMeasurement(
                        0,
                        ExpertHistogramSource::DecodeToken,
                        1,
                        1);
                    if (status ==
                        MoEOverlayServiceMeasurementRecordStatus::Recorded)
                    {
                        recorded.fetch_add(1, std::memory_order_relaxed);
                    }
                    else if (status ==
                             MoEOverlayServiceMeasurementRecordStatus::Contended)
                    {
                        contended.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
                        unexpected.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            });
    }
    start.store(true, std::memory_order_release);
    for (auto &worker : workers)
        worker.join();

    EXPECT_EQ(
        recorded.load() + contended.load(),
        static_cast<uint64_t>(kThreads * kCallsPerThread));
    EXPECT_EQ(unexpected.load(), 0u);
    EXPECT_EQ(endpoint.droppedServiceMeasurementCount(), contended.load());
    std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
    ASSERT_TRUE(endpoint.trySnapshotServiceMeasurements(&rows));
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].total_nanoseconds[0], recorded.load());
    EXPECT_EQ(rows[0].activation_count[0], recorded.load());
    EXPECT_EQ(rows[0].sample_count[0], recorded.load());

    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0,
            ExpertHistogramSource::PrefillChunk,
            std::numeric_limits<uint64_t>::max(),
            1),
        MoEOverlayServiceMeasurementRecordStatus::Recorded);
    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::PrefillChunk, 1, 1),
        MoEOverlayServiceMeasurementRecordStatus::Overflow);
    ASSERT_TRUE(endpoint.trySnapshotServiceMeasurements(&rows));
    EXPECT_FALSE(rows[0].valid());

    MoEOverlayParticipantResidency static_endpoint({
        .participant_id = 1,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 1,
    });
    EXPECT_EQ(
        static_endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::DecodeToken, 1, 1),
        MoEOverlayServiceMeasurementRecordStatus::Disabled);
    EXPECT_FALSE(static_endpoint.trySnapshotServiceMeasurements(&rows));
}
