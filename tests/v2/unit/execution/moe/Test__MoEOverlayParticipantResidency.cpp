/**
 * @file Test__MoEOverlayParticipantResidency.cpp
 * @brief Device-free RCU tests for participant-local prepared expert banks.
 */

#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayParticipantMigration.h"
#include "execution/moe/MoEOverlayFixedDownProjectionBank.h"
#include "execution/moe/MoERuntimeTable.h"
#include "execution/moe/DecodeExpertHistogram.h"

#include "loaders/ExpertGemmRegistry.h"
#include "tensors/TensorKernels.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <barrier>
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

    MoEOverlayPreparedExpertPayload payload(int base)
    {
        return {
            std::make_shared<IdentityGemm>(base + 1),
            std::make_shared<IdentityGemm>(base + 2),
            std::make_shared<IdentityGemm>(base + 3),
        };
    }

    MoEOverlayParticipantResidencyBank initialBank(
        const MoEOverlayPreparedExpertPayload &expert_zero)
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

    /** @brief Exercise the production prepare-then-publish bank lifecycle. */
    MoEOverlayParticipantBankInstallStatus prepareAndInstall(
        MoEOverlayParticipantResidency &residency,
        const MoEOverlayParticipantResidencyBank &bank,
        std::string *error)
    {
        auto prepared = residency.prepareReadyBank(bank, error);
        if (!prepared)
            return MoEOverlayParticipantBankInstallStatus::Invalid;
        return residency.installReadyBank(std::move(*prepared), error);
    }

    MoEExpertOwnerMap twoParticipantOwnerMap(int num_layers = 1)
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
        plan.placements.reserve(static_cast<std::size_t>(num_layers));
        for (int layer = 0; layer < num_layers; ++layer)
        {
            plan.placements.push_back({
                .layer = layer,
                .routed_expert_tier = {0, 1},
            });
        }
        return MoEExpertOwnerMap::build(plan);
    }

    /** @brief Construct a dynamic two-tier plan with one expert in each tier. */
    MoERoutedExpertPlacementPlan dynamicTwoParticipantPlan(
        bool cuda_continuation = false)
    {
        RoutedExpertDomain hot;
        hot.name = "hot_domain";
        hot.scope = ExecutionDomainScope::SINGLE;
        hot.backend = CollectiveBackendType::HOST;
        hot.participants = {
            cuda_continuation ? GlobalDeviceAddress::cuda(0, 0)
                              : GlobalDeviceAddress::cpu(0)};
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
                const auto engines = payload(
                    2000 + migration.destination.owner_participant * 100 +
                    migration.expert_id * 10);
                entry.projections[0] =
                    std::make_unique<MoEOverlayPreparedProjectionOperation>(
                        std::make_unique<ReadyProjectionTransfer>(),
                        entry.destination_arrival,
                        ExpertTierWeightProjection::Gate,
                        engines.gate());
                entry.projections[1] =
                    std::make_unique<MoEOverlayPreparedProjectionOperation>(
                        std::make_unique<ReadyProjectionTransfer>(),
                        entry.destination_arrival,
                        ExpertTierWeightProjection::Up,
                        engines.up());
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
                                                     engines.down()));
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

    /** @brief Script shared by a device-free GPU publication transaction. */
    struct ScriptedDeviceBankPublication
    {
        enum class Phase
        {
            Created,
            Preparing,
            Prepared,
            Publishing,
            Published,
            Aborting,
            Aborted,
            RetirementFencing,
            RetirementReady,
            Retired,
        };

        MoEOverlayResidencyWaveProgress prepare_progress =
            MoEOverlayResidencyWaveProgress::Pending;
        MoEOverlayResidencyWaveProgress publication_progress =
            MoEOverlayResidencyWaveProgress::Pending;
        MoEOverlayResidencyWaveProgress abort_progress =
            MoEOverlayResidencyWaveProgress::Pending;
        MoEOverlayResidencyWaveProgress retirement_progress =
            MoEOverlayResidencyWaveProgress::Pending;
        Phase phase = Phase::Created;
        std::uint64_t previous_epoch = 0;
        std::uint64_t candidate_epoch = 0;
        int create_calls = 0;
        int begin_prepare_calls = 0;
        int poll_prepare_calls = 0;
        int begin_publication_calls = 0;
        int poll_publication_calls = 0;
        int abort_calls = 0;
        int poll_abort_calls = 0;
        int poll_retirement_calls = 0;
        int retire_calls = 0;
    };

    /**
     * @brief Device-free implementation of the production inactive-bank ABI.
     *
     * Tests mutate only the four progress terminals. The fake otherwise
     * enforces the same irreversible publication and retirement ordering as
     * the CUDA/ROCm publisher, making lifecycle regressions deterministic.
     */
    class ScriptedDeviceBankTransaction final
        : public IMoEOverlayInactiveBankTransaction
    {
    public:
        explicit ScriptedDeviceBankTransaction(
            std::shared_ptr<ScriptedDeviceBankPublication> script)
            : script_(std::move(script))
        {
        }

        bool beginPrepare(std::string *error) noexcept override
        {
            if (error)
                error->clear();
            if (!script_ || script_->phase !=
                                ScriptedDeviceBankPublication::Phase::Created)
            {
                if (error)
                    *error = "scripted device preparation has invalid phase";
                return false;
            }
            ++script_->begin_prepare_calls;
            script_->phase =
                ScriptedDeviceBankPublication::Phase::Preparing;
            return true;
        }

        MoEOverlayResidencyWaveProgress pollPrepare(
            std::string *error) noexcept override
        {
            ++script_->poll_prepare_calls;
            if (script_->phase ==
                ScriptedDeviceBankPublication::Phase::Prepared)
                return MoEOverlayResidencyWaveProgress::Ready;
            if (script_->phase !=
                ScriptedDeviceBankPublication::Phase::Preparing)
            {
                if (error)
                    *error = "scripted device preparation is not in flight";
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            if (script_->prepare_progress ==
                MoEOverlayResidencyWaveProgress::Ready)
            {
                script_->phase =
                    ScriptedDeviceBankPublication::Phase::Prepared;
            }
            else if (script_->prepare_progress ==
                     MoEOverlayResidencyWaveProgress::Failed)
            {
                if (error)
                    *error = "scripted device preparation failed";
            }
            return script_->prepare_progress;
        }

        bool beginPublication(std::string *error) noexcept override
        {
            if (error)
                error->clear();
            if (script_->phase !=
                ScriptedDeviceBankPublication::Phase::Prepared)
            {
                if (error)
                    *error = "scripted device publication has invalid phase";
                return false;
            }
            ++script_->begin_publication_calls;
            script_->phase =
                ScriptedDeviceBankPublication::Phase::Publishing;
            return true;
        }

        MoEOverlayResidencyWaveProgress pollPublication(
            std::string *error) noexcept override
        {
            ++script_->poll_publication_calls;
            if (script_->phase ==
                ScriptedDeviceBankPublication::Phase::Published)
                return MoEOverlayResidencyWaveProgress::Ready;
            if (script_->phase !=
                ScriptedDeviceBankPublication::Phase::Publishing)
            {
                if (error)
                    *error = "scripted device publication is not in flight";
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            if (script_->publication_progress ==
                MoEOverlayResidencyWaveProgress::Ready)
            {
                script_->phase =
                    ScriptedDeviceBankPublication::Phase::Published;
            }
            else if (script_->publication_progress ==
                     MoEOverlayResidencyWaveProgress::Failed)
            {
                if (error)
                    *error = "scripted device publication failed";
            }
            return script_->publication_progress;
        }

        void abort() noexcept override
        {
            ++script_->abort_calls;
            if (script_->phase ==
                    ScriptedDeviceBankPublication::Phase::Publishing ||
                script_->phase ==
                    ScriptedDeviceBankPublication::Phase::Published)
            {
                std::terminate();
            }
            script_->phase =
                ScriptedDeviceBankPublication::Phase::Aborting;
        }

        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept override
        {
            ++script_->poll_abort_calls;
            if (script_->phase ==
                ScriptedDeviceBankPublication::Phase::Aborted)
                return MoEOverlayResidencyWaveProgress::Ready;
            if (script_->phase !=
                ScriptedDeviceBankPublication::Phase::Aborting)
            {
                if (error)
                    *error = "scripted device abort is not in flight";
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            if (script_->abort_progress ==
                MoEOverlayResidencyWaveProgress::Ready)
            {
                script_->phase =
                    ScriptedDeviceBankPublication::Phase::Aborted;
            }
            return script_->abort_progress;
        }

        MoEOverlayResidencyWaveProgress pollRetirementFence(
            std::string *error) noexcept override
        {
            ++script_->poll_retirement_calls;
            if (script_->phase ==
                ScriptedDeviceBankPublication::Phase::RetirementReady)
                return MoEOverlayResidencyWaveProgress::Ready;
            if (script_->phase ==
                ScriptedDeviceBankPublication::Phase::Published)
            {
                script_->phase = ScriptedDeviceBankPublication::Phase::
                    RetirementFencing;
            }
            if (script_->phase != ScriptedDeviceBankPublication::Phase::
                                      RetirementFencing)
            {
                if (error)
                    *error = "scripted device retirement has invalid phase";
                return MoEOverlayResidencyWaveProgress::Failed;
            }
            if (script_->retirement_progress ==
                MoEOverlayResidencyWaveProgress::Ready)
            {
                script_->phase = ScriptedDeviceBankPublication::Phase::
                    RetirementReady;
            }
            return script_->retirement_progress;
        }

        void retirePrevious() noexcept override
        {
            if (script_->phase != ScriptedDeviceBankPublication::Phase::
                                      RetirementReady)
                std::terminate();
            ++script_->retire_calls;
            script_->phase =
                ScriptedDeviceBankPublication::Phase::Retired;
        }

    private:
        std::shared_ptr<ScriptedDeviceBankPublication> script_;
    };

    /** @brief Factory fake that captures exact epoch identities. */
    class ScriptedDeviceBankPublisher final
        : public IMoEOverlayHostAuthorityDeviceBankPublisher
    {
    public:
        explicit ScriptedDeviceBankPublisher(
            std::shared_ptr<ScriptedDeviceBankPublication> script)
            : script_(std::move(script))
        {
        }

        std::unique_ptr<IMoEOverlayInactiveBankTransaction>
        createTransaction(
            std::shared_ptr<const MoEOverlayResidencySnapshot> previous,
            std::shared_ptr<const MoEOverlayResidencySnapshot> candidate,
            std::string *error) noexcept override
        {
            if (error)
                error->clear();
            if (!script_ || !previous || !candidate ||
                candidate->epoch != previous->epoch + 1u ||
                script_->create_calls != 0)
            {
                if (error)
                    *error = "scripted device publisher rejected transaction";
                return nullptr;
            }
            ++script_->create_calls;
            script_->previous_epoch = previous->epoch;
            script_->candidate_epoch = candidate->epoch;
            return std::make_unique<ScriptedDeviceBankTransaction>(script_);
        }

    private:
        std::shared_ptr<ScriptedDeviceBankPublication> script_;
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
    ParticipantMigrationFixture participantMigrationFixture(
        bool cuda_continuation = false)
    {
        ParticipantMigrationFixture fixture;
        const auto plan = dynamicTwoParticipantPlan(cuda_continuation);
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = 2;
        histogram_config.top_k = 1;
        histogram_config.window_size = 2;
        histogram_config.sockets = {
            cuda_continuation ? DeviceId::cuda(0) : DeviceId::cpu(),
            DeviceId::cpu()};
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
                .perf_device = cuda_continuation ? "cuda/cpu" :
                                                   "cpu/cpu",
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
            std::vector<MoEOverlayPreparedExpertPayload> engines(2);
            for (int expert = 0; expert < 2; ++expert)
            {
                if (mask[static_cast<std::size_t>(expert)])
                {
                    engines[static_cast<std::size_t>(expert)] =
                        payload(1700 + participant_id * 100 + expert * 10);
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

/** @brief Payloads cannot represent partial families or leave false-ready moved-from values. */
TEST(MoEOverlayPreparedPayload, ExactFamiliesAndMoveOwnership)
{
    using Payload = MoEOverlayPreparedExpertPayload;
    auto whole = payload(7000);
    auto pair = Payload::gateUp(whole.gate(), whole.up());
    ASSERT_TRUE(pair.readyFor(DeviceMoEProjectionSet::GateUp));
    EXPECT_FALSE(pair.complete());
    EXPECT_EQ(pair.down(), nullptr);
    EXPECT_FALSE(pair.sameIdentity(whole));
    EXPECT_THROW(Payload(whole.gate(), whole.up(), {}), std::invalid_argument);
    EXPECT_THROW(Payload::gateUp({}, whole.up()), std::invalid_argument);
    EXPECT_THROW(Payload::fromProjections(DeviceMoEProjectionSet::GateUp,
        {whole.gate(), whole.up(), whole.down()}), std::invalid_argument);
    EXPECT_THROW(Payload::fromProjections(static_cast<DeviceMoEProjectionSet>(99),
        {whole.gate(), whole.up(), {}}), std::invalid_argument);

    auto retained = pair;
    auto moved = std::move(pair);
    EXPECT_TRUE(moved.sameIdentity(retained));
    EXPECT_TRUE(pair.empty());
    EXPECT_EQ(pair.gate(), nullptr);
    EXPECT_THROW((void)pair.projections(), std::logic_error);
    pair = std::move(moved);
    EXPECT_TRUE(moved.empty());
    EXPECT_TRUE(pair.sameIdentity(retained));
    auto *same = &pair;
    pair = std::move(*same);
    EXPECT_TRUE(pair.sameIdentity(retained));
}

/** @brief A two-projection arrival never waits for or accepts immutable down weights. */
TEST(MoEOverlayPreparedPayload, GateUpArrivalIsAtomicAndRejectsFixedDown)
{
    const auto whole = payload(7100);
    MoEOverlayPreparedExpertArrival arrival(DeviceMoEProjectionSet::GateUp);
    std::string error;
    MoEOverlayPreparedExpertPayload result = whole;
    EXPECT_FALSE(arrival.completePayload(result, &error));
    EXPECT_TRUE(result.empty());
    EXPECT_FALSE(arrival.publish(ExpertTierWeightProjection::Down, whole.down(), &error));
    ASSERT_TRUE(arrival.publish(ExpertTierWeightProjection::Gate, whole.gate(), &error));
    EXPECT_FALSE(arrival.completePayload(result, &error));
    EXPECT_TRUE(result.empty());
    EXPECT_FALSE(arrival.publish(ExpertTierWeightProjection::Gate, payload(7200).gate(), &error));
    ASSERT_TRUE(arrival.publish(ExpertTierWeightProjection::Up, whole.up(), &error));
    ASSERT_TRUE(arrival.completePayload(result, &error)) << error;
    EXPECT_TRUE(result.readyFor(DeviceMoEProjectionSet::GateUp));
    EXPECT_FALSE(result.complete());
    EXPECT_EQ(result.down(), nullptr);
    EXPECT_TRUE(arrival.fail("late transport failure"));
    EXPECT_FALSE(arrival.completePayload(result, &error));
    EXPECT_TRUE(result.empty());
}

/** @brief Independent transfer completions compose one immutable family under contention. */
TEST(MoEOverlayPreparedPayload, ConcurrentGateUpArrivalPublication)
{
    for (int iteration = 0; iteration < 20; ++iteration)
    {
        SCOPED_TRACE(iteration);
        const auto whole = payload(7300 + iteration * 3);
        MoEOverlayPreparedExpertArrival arrival(DeviceMoEProjectionSet::GateUp);
        std::barrier start(3);
        std::atomic<bool> accepted{true};
        std::jthread gate([&] {
            start.arrive_and_wait();
            if (!arrival.publish(ExpertTierWeightProjection::Gate, whole.gate())) accepted = false;
        });
        std::jthread up([&] {
            start.arrive_and_wait();
            if (!arrival.publish(ExpertTierWeightProjection::Up, whole.up())) accepted = false;
        });
        start.arrive_and_wait();
        gate.join();
        up.join();
        ASSERT_TRUE(accepted);
        MoEOverlayPreparedExpertPayload result;
        ASSERT_TRUE(arrival.completePayload(result));
        EXPECT_EQ(result.gate(), whole.gate());
        EXPECT_EQ(result.up(), whole.up());
        EXPECT_EQ(result.down(), nullptr);
    }
}

/** @brief The existing RCU mechanism retains and retires pairs without changing their contract. */
TEST(MoEOverlayPreparedPayload, GateUpEpochsRetainReadersAndRejectContractChanges)
{
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        // Device identifiers are metadata only: no backend is initialized.
        MoEOverlayParticipantResidency residency({
            .participant_id = 4, .device = device, .num_layers = 1, .num_experts = 2,
            .movable_projections = DeviceMoEProjectionSet::GateUp});
        MoEOverlayParticipantResidencyBank bank;
        bank.epoch = 1;
        bank.participant_id = 4;
        bank.device = device;
        bank.layers.resize(1);
        auto &layer = bank.layers[0];
        layer.resident_mask.assign(2, false);
        layer.experts.resize(2);
        layer.movable_projections = DeviceMoEProjectionSet::GateUp;
        auto whole = payload(7400);
        std::weak_ptr<ITensorGemm> old_gate = whole.gate();
        layer.setResidentExpert(0, MoEOverlayPreparedExpertPayload::gateUp(whole.gate(), whole.up()));
        whole = {};
        std::string error;
        ASSERT_EQ(prepareAndInstall(residency, bank, &error), MoEOverlayParticipantBankInstallStatus::Installed) << error;
        auto reader = residency.acquire(1);
        ASSERT_TRUE(reader);
        auto candidate = residency.cloneCandidate(1, 2);
        candidate.layers[0].clearExpert(0);
        auto next = payload(7500);
        EXPECT_THROW(candidate.layers[0].setResidentExpert(1, next), std::invalid_argument);
        candidate.layers[0].setResidentExpert(1, MoEOverlayPreparedExpertPayload::gateUp(next.gate(), next.up()));
        ASSERT_EQ(prepareAndInstall(residency, candidate, &error), MoEOverlayParticipantBankInstallStatus::Installed) << error;
        bank = {};
        ASSERT_TRUE(residency.retire(1));
        EXPECT_FALSE(old_gate.expired());
        reader.reset();
        EXPECT_FALSE(residency.retire(99)); // Maintenance reclaims the drained old node.
        EXPECT_TRUE(old_gate.expired());

        auto forged = residency.cloneCandidate(2, 3);
        forged.layers[0].clearExpert(1);
        forged.layers[0].movable_projections = DeviceMoEProjectionSet::CompleteExpert;
        // Even an empty layer cannot change the endpoint's physical contract.
        EXPECT_FALSE(residency.prepareReadyBank(forged, &error));
        EXPECT_NE(error.find("incomplete"), std::string::npos);
        EXPECT_TRUE(residency.acquire(2));
    }
    EXPECT_THROW((MoEOverlayParticipantResidency{
        {.participant_id = 0, .device = DeviceId::cpu(), .num_layers = 1, .num_experts = 2,
         .movable_projections = DeviceMoEProjectionSet::GateUp}}), std::invalid_argument);
}

/** @test Fixed down slices outlive registry reset and do not follow movable ownership. */
TEST(MoEOverlayPreparedPayload, FixedDownLifetimeIsIndependentOfMovableResidency)
{
    for (const auto device : {DeviceId::cuda(3), DeviceId::rocm(3)})
    {
        // Logical slice 1 deliberately differs from the physical device ID.
        const MoEExpertOwnerParticipant participant{
            .participant_id = 7, .domain_name = "compute", .domain_participant_index = 1,
            .device = device, .world_rank = 2, .world_rank_known = true};
        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({4, 256, 512}, 1, 2);
        ExpertGemmRegistry registry;
        std::vector<std::weak_ptr<ITensorGemm>> down_lifetimes;
        const auto register_role = [&](int expert, ExpertGemmRegistry::WeightRole role)
        {
            auto engine = std::make_shared<IdentityGemm>(100 * expert + static_cast<int>(role));
            registry.registerEngineForParticipant("compute", device, 2, 1, 0, expert,
                role, engine.get(), engine, ownership);
            return std::weak_ptr<ITensorGemm>(engine);
        };
        for (int expert = 0; expert < 4; ++expert)
            down_lifetimes.push_back(register_role(expert, ExpertGemmRegistry::WeightRole::DOWN));
        auto fixed = MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0, ownership);
        ASSERT_EQ(fixed->engines().size(), 4u);
        EXPECT_EQ(fixed->ownership(), ownership);
        EXPECT_EQ(fixed->participantId(), 7);
        EXPECT_EQ(fixed->device(), device);
        EXPECT_TRUE(fixed->sameIdentity(*MoEOverlayFixedDownProjectionBank::resolve(
            registry, participant, 0, ownership)));

        std::vector<MoEOverlayPreparedExpertPayload> movable;
        std::string error;
        // A participant owning no gate/up pairs still needs every fixed down slice.
        ASSERT_TRUE(resolveMoEOverlayPreparedExpertPayloads(registry, participant, 0, 4,
            {false, false, false, false}, movable, &error, ownership)) << error;
        for (const auto &entry : movable) EXPECT_TRUE(entry.empty());
        const auto gate_lifetime = register_role(2, ExpertGemmRegistry::WeightRole::GATE);
        register_role(2, ExpertGemmRegistry::WeightRole::UP);
        ASSERT_TRUE(resolveMoEOverlayPreparedExpertPayloads(registry, participant, 0, 4,
            {false, false, true, false}, movable, &error, ownership)) << error;
        EXPECT_TRUE(movable[2].readyFor(DeviceMoEProjectionSet::GateUp));
        EXPECT_FALSE(movable[2].down());
        EXPECT_FALSE(movable[2].complete());

        // Registry retirement cannot release weights embedded in a live graph.
        registry.clear();
        EXPECT_FALSE(gate_lifetime.expired());
        movable.clear();
        EXPECT_TRUE(gate_lifetime.expired());
        for (const auto &lifetime : down_lifetimes) EXPECT_FALSE(lifetime.expired());
        fixed.reset();
        for (const auto &lifetime : down_lifetimes) EXPECT_TRUE(lifetime.expired());
    }
}

/** @test An incomplete or differently scoped fixed inventory cannot bind a graph. */
TEST(MoEOverlayPreparedPayload, FixedDownRequiresCompleteExactIdentity)
{
    ExpertGemmRegistry registry;
    const MoEExpertOwnerParticipant participant{
        .participant_id = 7, .domain_name = "compute", .domain_participant_index = 1,
        .device = DeviceId::rocm(3), .world_rank = 2, .world_rank_known = true};
    const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({2, 256, 512}, 1, 2);
    auto engine = std::make_shared<IdentityGemm>(5);
    registry.registerEngineForParticipant("compute", participant.device, 2, 1, 0, 0,
        ExpertGemmRegistry::WeightRole::DOWN, engine.get(), engine, ownership);
    EXPECT_THROW(MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0, ownership), std::runtime_error);
    registry.registerEngineForParticipant("compute", participant.device, 2, 1, 0, 1,
        ExpertGemmRegistry::WeightRole::DOWN, engine.get(), engine, ownership);
    auto fixed = MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0, ownership);
    auto other_rank = participant;
    other_rank.world_rank = 3;
    EXPECT_THROW(MoEOverlayFixedDownProjectionBank::resolve(registry, other_rank, 0, ownership), std::runtime_error);
    auto wrong_device = participant;
    wrong_device.device = DeviceId::cpu();
    EXPECT_THROW(MoEOverlayFixedDownProjectionBank::resolve(registry, wrong_device, 0, ownership), std::invalid_argument);
    EXPECT_THROW(MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0,
        MoEExpertProjectionOwnership::completeExperts({2, 256, 512}, 1, 2)), std::invalid_argument);
    EXPECT_THROW(MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0,
        MoEExpertProjectionOwnership::gateUpOwnedDownColumns({2, 256, 512}, 0, 2)), std::invalid_argument);
    EXPECT_THROW(MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0,
        MoEExpertProjectionOwnership::gateUpOwnedDownColumns({2, 256, 256}, 1, 2)), std::logic_error);

    // Equal geometry is not equal capture identity after an engine replacement.
    auto replacement = std::make_shared<IdentityGemm>(6);
    registry.registerEngineForParticipant("compute", participant.device, 2, 1, 0, 1,
        ExpertGemmRegistry::WeightRole::DOWN, replacement.get(), replacement, ownership);
    EXPECT_FALSE(fixed->sameIdentity(*MoEOverlayFixedDownProjectionBank::resolve(
        registry, participant, 0, ownership)));
}

/** @test Runtime publication and portable restore preserve the immutable projection contract. */
TEST(MoEOverlayPreparedPayload, RuntimeResetAndPrefixRestoreRetainFixedDownBindings)
{
    for (const auto device : {DeviceId::cuda(3), DeviceId::rocm(3)})
    for (const auto format : {DeviceMoEWeightFormat::NativeVNNI, DeviceMoEWeightFormat::FP16,
                             DeviceMoEWeightFormat::BF16, DeviceMoEWeightFormat::FP32})
    {
        // These are metadata-only device tables. No GPU driver or engine is invoked.
        ExpertGemmRegistry registry;
        const MoEExpertOwnerParticipant participant{
            .participant_id = 7, .domain_name = "compute", .domain_participant_index = 1,
            .device = device, .world_rank = 2, .world_rank_known = true};
        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns({2, 256, 512}, 1, 2);
        for (int expert = 0; expert < 2; ++expert)
        {
            auto engine = std::make_shared<IdentityGemm>(expert);
            registry.registerEngineForParticipant("compute", device, 2, 1, 0, expert,
                ExpertGemmRegistry::WeightRole::DOWN, engine.get(), engine, ownership);
        }
        auto fixed = MoEOverlayFixedDownProjectionBank::resolve(registry, participant, 0, ownership);
        DeviceMoERuntimeTable::Config config{.device_id = device, .num_layers = 1,
            .num_experts = 2, .top_k = 1, .fixed_down_banks = {fixed}};
        DeviceMoERuntimeTable table(config);
        EXPECT_EQ(table.fixedDownProjectionBank(0), fixed.get());
        EXPECT_EQ(table.movableProjections(), DeviceMoEProjectionSet::GateUp);
        EXPECT_THROW((void)table.fixedDownProjectionBank(1), std::out_of_range);

        MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = 2;
        update.participant_id = 1;
        update.participant_count = 2;
        update.experts.resize(2);
        update.local_compute_mask = {0, 1};
        update.replica_role = {static_cast<uint8_t>(DeviceMoEReplicaRole::None),
                              static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)};
        update.resident_participant_mask = {1, 2};
        for (int expert = 0; expert < 2; ++expert)
        {
            auto &descriptor = update.experts[expert];
            descriptor.logical_expert_id = expert;
            descriptor.owner_participant = expert;
            descriptor.projection_set = DeviceMoEProjectionSet::GateUp;
        }
        auto &local = update.experts[1];
        local.local_slot = 1;
        local.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
            DeviceMoEExpertFlags::Resident | DeviceMoEExpertFlags::LocalCompute);
        local.weight_format = format;
        if (format == DeviceMoEWeightFormat::NativeVNNI)
        {
            local.gate.payload = reinterpret_cast<const uint8_t *>(0x1000u);
            local.gate.scales = reinterpret_cast<const void *>(0x2000u);
            local.gate.n = 512;
            local.gate.k = 256;
            local.gate.blocks_per_row = 8;
            local.gate.codebook_id = 7;
            local.up = local.gate;
        }
        else
        {
            local.floating_gate = {reinterpret_cast<const void *>(0x1000u), 512, 256};
            local.floating_up = {reinterpret_cast<const void *>(0x2000u), 512, 256};
        }
        ASSERT_TRUE(local.movableWeightsReady());
        ASSERT_TRUE(table.prepareInactiveBank(0, update));
        ASSERT_TRUE(table.flipActiveBank(0, 1, nullptr));
        std::vector<DeviceMoEPortableLayerRuntimeState> saved;
        ASSERT_TRUE(table.capturePortableRuntimeState(saved));
        table.resetDecodeRuntimeState();
        EXPECT_EQ(table.fixedDownProjectionBank(0), fixed.get());
        ASSERT_TRUE(table.restorePortableRuntimeState(saved, nullptr,
            [&](int layer, int expert, int slot, DeviceMoEExpertDescriptor &descriptor)
            {
                EXPECT_EQ(layer, 0);
                EXPECT_EQ(expert, 1);
                EXPECT_EQ(slot, 1);
                descriptor = local;
                return true;
            }));
        const auto &state = table.hostLayerState(0);
        EXPECT_EQ(state.banks[state.active_bank].experts[1].projection_set, DeviceMoEProjectionSet::GateUp);
        EXPECT_FALSE(state.banks[state.active_bank].experts[1].weightsReady());
        EXPECT_EQ(table.fixedDownProjectionBank(0), fixed.get());
        auto raw_snapshot = std::make_unique<DeviceMoELayerRuntime>(state);
        const auto accepted_epoch = state.active_epoch;
        raw_snapshot->banks[raw_snapshot->active_bank].experts[1].projection_set = DeviceMoEProjectionSet::CompleteExpert;
        EXPECT_THROW(table.restoreRuntimeStateSnapshot(raw_snapshot.get(), 1), std::invalid_argument);
        EXPECT_EQ(table.hostLayerState(0).active_epoch, accepted_epoch);
        *raw_snapshot = state;
        raw_snapshot->participant_count = 4;
        EXPECT_THROW(table.restoreRuntimeStateSnapshot(raw_snapshot.get(), 1), std::invalid_argument);
        EXPECT_EQ(table.fixedDownProjectionBank(0), fixed.get());
        *raw_snapshot = state;
        raw_snapshot->banks[raw_snapshot->active_bank].epoch = 0;
        EXPECT_THROW(table.restoreRuntimeStateSnapshot(raw_snapshot.get(), 1), std::invalid_argument);

        table.resetDecodeRuntimeState();
        // A stale full-FFN resolver is not a permissible rehydration of a pair.
        EXPECT_FALSE(table.restorePortableRuntimeState(saved, nullptr,
            [&](int, int, int, DeviceMoEExpertDescriptor &descriptor)
            {
                descriptor = local;
                descriptor.projection_set = DeviceMoEProjectionSet::CompleteExpert;
                return true;
            }));
        table.restoreInitialRuntimeState();

        // An epoch may change owners, never the captured arithmetic partition.
        auto invalid = update;
        invalid.epoch = table.hostLayerState(0).active_epoch + 1;
        invalid.experts[1].projection_set = DeviceMoEProjectionSet::CompleteExpert;
        EXPECT_THROW(table.prepareInactiveBank(0, invalid), std::invalid_argument);
        invalid = update;
        invalid.epoch = table.hostLayerState(0).active_epoch + 1;
        invalid.participant_count = 4;
        EXPECT_THROW(table.prepareInactiveBank(0, invalid), std::invalid_argument);
        invalid = update;
        invalid.epoch = table.hostLayerState(0).active_epoch + 1;
        // Even an inactive/remote descriptor must never carry fixed down bytes.
        invalid.experts[0].floating_down = {reinterpret_cast<const void *>(0x3000u), 128, 512};
        EXPECT_THROW(table.prepareInactiveBank(0, invalid), std::invalid_argument);

        // Conversely, a whole-expert table cannot acquire a partial family.
        DeviceMoERuntimeTable whole(device, 1, 2, 1, false);
        EXPECT_THROW(whole.prepareInactiveBank(0, update), std::invalid_argument);
        auto invalid_config = config;
        invalid_config.fixed_down_banks = {nullptr};
        EXPECT_THROW((DeviceMoERuntimeTable{invalid_config}), std::invalid_argument);
        invalid_config = config;
        invalid_config.device_id = DeviceId::cpu();
        EXPECT_THROW((DeviceMoERuntimeTable{invalid_config}), std::invalid_argument);
        invalid_config = config;
        invalid_config.num_layers = 2;
        EXPECT_THROW((DeviceMoERuntimeTable{invalid_config}), std::invalid_argument);
    }
}

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

    const auto old_expert = payload(100);
    auto epoch_one = initialBank(old_expert);
    std::string error;
    EXPECT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    auto epoch_two = residency.cloneCandidate(1, 2);
    epoch_two.layers[0].clearExpert(0);
    const auto new_expert = payload(200);
    epoch_two.layers[0].setResidentExpert(1, new_expert);
    EXPECT_EQ(
        prepareAndInstall(residency, epoch_two, &error),
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
        acquired_one->layers[0].experts[0].gate().get(),
        old_expert.gate().get());
    EXPECT_EQ(
        acquired_two->layers[0].experts[1].gate().get(),
        new_expert.gate().get());
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
    auto epoch_one = initialBank(payload(300));
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    auto epoch_two = residency.cloneCandidate(1, 2);
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);

    auto epoch_three = residency.cloneCandidate(2, 3);
    EXPECT_FALSE(residency.hasCandidateCapacity());
    EXPECT_EQ(
        prepareAndInstall(residency, epoch_three, &error),
        MoEOverlayParticipantBankInstallStatus::CapacityUnavailable);
    EXPECT_EQ(residency.retainedEpochCount(), 2u);

    EXPECT_TRUE(residency.retire(1));
    EXPECT_TRUE(residency.hasCandidateCapacity());
    EXPECT_EQ(
        prepareAndInstall(residency, epoch_three, &error),
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
    auto old_expert = payload(400);
    std::weak_ptr<ITensorGemm> old_gate = old_expert.gate();
    auto epoch_one = initialBank(old_expert);
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    auto epoch_two = residency.cloneCandidate(1, 2);
    epoch_two.layers[0].clearExpert(0);
    epoch_two.layers[0].setResidentExpert(1, payload(500));
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);

    /* Drop every construction-time copy; epoch one is now the only owner. */
    old_expert = {};
    epoch_one = {};
    EXPECT_FALSE(old_gate.expired());

    EXPECT_TRUE(residency.retire(1));
    EXPECT_TRUE(old_gate.expired());
}

TEST(Test__MoEOverlayParticipantResidency,
     AcquiredLeaseSurvivesConcurrentRetirementUntilReaderReleases)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });
    std::string error;
    auto old_expert = payload(550);
    std::weak_ptr<ITensorGemm> old_gate = old_expert.gate();
    auto epoch_one = initialBank(old_expert);
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    auto inference_lease = residency.acquire(1);
    ASSERT_NE(inference_lease, nullptr);
    old_expert = {};
    epoch_one = {};

    /* Unlink is immediate, but reclamation must respect the live reader. */
    ASSERT_TRUE(residency.retire(1));
    EXPECT_EQ(residency.acquire(1), nullptr);
    EXPECT_FALSE(old_gate.expired());
    EXPECT_EQ(inference_lease->epoch, 1u);
    EXPECT_TRUE(inference_lease->layers[0].resident_mask[0]);

    inference_lease.reset();
    /* Any later maintenance visit performs deferred destruction off-path. */
    EXPECT_FALSE(residency.retire(999));
    EXPECT_TRUE(old_gate.expired());
}

TEST(Test__MoEOverlayParticipantResidency,
     InferenceReadersProgressWhileMaintenanceCyclesCandidateSlots)
{
    MoEOverlayParticipantResidency residency({
        .participant_id = 4,
        .device = DeviceId::cpu(),
        .num_layers = 1,
        .num_experts = 2,
        .retained_epoch_capacity = 2,
    });
    std::string error;
    const auto epoch_one = initialBank(payload(575));
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;

    std::atomic<bool> stop{false};
    std::atomic<bool> reader_failed{false};
    std::atomic<uint64_t> reader_acquisitions{0};
    constexpr int kReaderCount = 4;
    constexpr uint64_t kMinimumMaintenanceCycles = 500;
    constexpr uint64_t kMaximumMaintenanceCycles = 10000;
    constexpr uint64_t kRequiredReaderAcquisitions = 1001;
    std::barrier start_gate(kReaderCount + 1);
    std::vector<std::thread> readers;
    readers.reserve(kReaderCount);
    for (int reader = 0; reader < kReaderCount; ++reader)
    {
        readers.emplace_back(
            [&]
            {
                /* Begin all lock-free readers with the maintenance writer. */
                start_gate.arrive_and_wait();
                while (!stop.load(std::memory_order_acquire))
                {
                    auto lease = residency.acquire(1);
                    if (!lease || lease->epoch != 1 ||
                        !lease->layers[0].resident_mask[0])
                    {
                        reader_failed.store(true, std::memory_order_release);
                        break;
                    }
                    reader_acquisitions.fetch_add(
                        1, std::memory_order_relaxed);
                }
            });
    }

    start_gate.arrive_and_wait();
    bool writer_failed = false;
    uint64_t maintenance_cycles = 0;
    while (maintenance_cycles < kMaximumMaintenanceCycles &&
           (maintenance_cycles < kMinimumMaintenanceCycles ||
            reader_acquisitions.load(std::memory_order_relaxed) <
                kRequiredReaderAcquisitions))
    {
        const uint64_t epoch = maintenance_cycles + 2;
        auto candidate = residency.cloneCandidate(1, epoch);
        auto prepared = residency.prepareReadyBank(
            std::move(candidate), &error);
        if (!prepared ||
            residency.installReadyBank(std::move(*prepared), &error) !=
                MoEOverlayParticipantBankInstallStatus::Installed ||
            !residency.retire(epoch))
        {
            writer_failed = true;
            break;
        }
        ++maintenance_cycles;

        /* Yield only in this adversarial unit test so a saturated test runner
         * cannot finish every tiny maintenance cycle before scheduling a
         * reader. Production publication itself remains lock-free and wait-free
         * with respect to the maintenance mutex. */
        std::this_thread::yield();
    }

    stop.store(true, std::memory_order_release);
    for (auto &reader : readers)
        reader.join();

    EXPECT_FALSE(writer_failed) << error;
    EXPECT_FALSE(reader_failed.load(std::memory_order_acquire));
    EXPECT_GE(maintenance_cycles, kMinimumMaintenanceCycles);
    EXPECT_GE(
        reader_acquisitions.load(std::memory_order_relaxed),
        kRequiredReaderAcquisitions);
    EXPECT_NE(residency.acquire(1), nullptr);
    EXPECT_EQ(residency.retainedEpochCount(), 1u);
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
    const auto epoch_one = initialBank(payload(600));
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    EXPECT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::AlreadyInstalled);

    auto conflicting = epoch_one;
    conflicting.layers[0].clearExpert(0);
    conflicting.layers[0].setResidentExpert(1, payload(700));
    EXPECT_EQ(
        prepareAndInstall(residency, conflicting, &error),
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
    auto invalid = initialBank(payload(800));
    invalid.layers[0].resident_mask[1] = true;
    // A partial family cannot be constructed. A resident mask over an empty
    // slot must still be rejected at the bank publication boundary.

    std::string error;
    EXPECT_EQ(
        prepareAndInstall(residency, invalid, &error),
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
    const auto epoch_one = initialBank(payload(1000));
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_one, &error),
        MoEOverlayParticipantBankInstallStatus::Installed);
    const auto epoch_two = residency.cloneCandidate(1, 2);
    ASSERT_EQ(
        prepareAndInstall(residency, epoch_two, &error),
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
    EXPECT_THROW(
        (void)registry.initialBankExpertSelections(),
        std::logic_error);
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

    std::vector<MoEOverlayPreparedExpertPayload> first_engines(2);
    first_engines[0] = payload(1100);
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

    std::vector<MoEOverlayPreparedExpertPayload> second_engines(2);
    second_engines[1] = payload(1200);
    EXPECT_TRUE(registry.registerInitialLayer(
        1,
        0,
        {false, true},
        second_engines,
        &error))
        << error;
    EXPECT_TRUE(registry.allInitialBanksReady());
    EXPECT_TRUE(registry.incompleteInitialBanks().empty());

    const auto selections = registry.initialBankExpertSelections();
    ASSERT_EQ(selections.size(), 2u);
    EXPECT_EQ(selections[0].participant_id, 0);
    EXPECT_EQ(selections[0].device, DeviceId::cpu());
    EXPECT_EQ(selections[0].layer_idx, 0);
    EXPECT_EQ(selections[0].expert_ids, (std::vector<int>{0}));
    EXPECT_TRUE(selections[0].matches_frozen_owner_map);
    EXPECT_EQ(selections[1].participant_id, 1);
    EXPECT_EQ(selections[1].layer_idx, 0);
    EXPECT_EQ(selections[1].expert_ids, (std::vector<int>{1}));
    EXPECT_TRUE(selections[1].matches_frozen_owner_map);

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
    std::vector<MoEOverlayPreparedExpertPayload> engines(2);
    engines[0] = payload(1300);
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
    different[0] = payload(1400);
    EXPECT_FALSE(registry.registerInitialLayer(
        0,
        0,
        {true, false},
        different,
        &error));
    EXPECT_FALSE(error.empty());
}

TEST(Test__MoEOverlayParticipantResidencyRegistry,
     LateGraphRegistrationCannotResurrectRetiredInitialEpoch)
{
    MoEOverlayParticipantResidencyRegistry registry({
        .owner_map = twoParticipantOwnerMap(),
        .local_participant_ids = {0},
        .num_layers = 1,
        .num_experts = 2,
        .initial_epoch = 1,
        .retained_epoch_capacity = 2,
    });
    std::vector<MoEOverlayPreparedExpertPayload> engines(2);
    engines[0] = payload(1450);
    std::string error;
    ASSERT_TRUE(registry.registerInitialLayer(
        0,
        0,
        {true, false},
        engines,
        &error))
        << error;

    const auto endpoint = registry.endpoint(0);
    ASSERT_NE(endpoint, nullptr);
    auto epoch_two = endpoint->cloneCandidate(1, 2);
    ASSERT_EQ(
        prepareAndInstall(*endpoint, epoch_two, &error),
        MoEOverlayParticipantBankInstallStatus::Installed)
        << error;
    ASSERT_TRUE(endpoint->retire(1));
    ASSERT_EQ(endpoint->acquire(1), nullptr);
    ASSERT_NE(endpoint->acquire(2), nullptr);
    ASSERT_TRUE(endpoint->hasCandidateCapacity());

    /*
     * A graph variant may validate the canonical bootstrap identities after
     * maintenance advances.  It must not republish epoch one or consume the
     * only inactive bank needed by epoch three.
     */
    EXPECT_TRUE(registry.registerInitialLayer(
        0,
        0,
        {true, false},
        engines,
        &error))
        << error;
    EXPECT_TRUE(registry.allInitialBanksReady());
    EXPECT_EQ(endpoint->acquire(1), nullptr);
    EXPECT_NE(endpoint->acquire(2), nullptr);
    EXPECT_EQ(endpoint->retainedEpochCount(), 1u);
    EXPECT_TRUE(endpoint->hasCandidateCapacity());
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
     ResolvesExactOwnedParticipantPayloadsAndRejectsPartialRoles)
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

    std::vector<MoEOverlayPreparedExpertPayload> resolved;
    std::string error;
    ASSERT_TRUE(resolveMoEOverlayPreparedExpertPayloads(
        engines,
        *participant,
        0,
        2,
        {true, false},
        resolved,
        &error))
        << error;
    ASSERT_EQ(resolved.size(), 2u);
    EXPECT_EQ(resolved[0].gate().get(), gate.get());
    EXPECT_EQ(resolved[0].up().get(), up.get());
    EXPECT_EQ(resolved[0].down().get(), down.get());
    EXPECT_TRUE(resolved[1].empty());

    EXPECT_FALSE(resolveMoEOverlayPreparedExpertPayloads(
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

TEST(Test__MoEOverlayParticipantResidencyRegistry,
     FinalizesDormantLayersFromModelOwnedPreparedEngines)
{
    const auto owner_map = twoParticipantOwnerMap(2);
    const auto *participant = owner_map.participantForId(0);
    ASSERT_NE(participant, nullptr);
    MoEOverlayParticipantResidencyRegistry residency({
        .owner_map = owner_map,
        .local_participant_ids = {0},
        .num_layers = 2,
        .num_experts = 2,
        .initial_epoch = 1,
    });
    ExpertGemmRegistry prepared_engines;
    const int world_rank = participant->world_rank_known
                               ? participant->world_rank
                               : -1;
    const auto register_payload =
        [&](int layer_idx,
            const MoEOverlayPreparedExpertPayload &engines)
    {
        const auto register_role =
            [&](ExpertGemmRegistry::WeightRole role,
                const std::shared_ptr<ITensorGemm> &engine)
        {
            prepared_engines.registerEngineForParticipant(
                participant->domain_name,
                participant->device,
                world_rank,
                participant->domain_participant_index,
                layer_idx,
                0,
                role,
                engine.get(),
                engine);
        };
        register_role(ExpertGemmRegistry::WeightRole::GATE, engines.gate());
        register_role(ExpertGemmRegistry::WeightRole::UP, engines.up());
        register_role(ExpertGemmRegistry::WeightRole::DOWN, engines.down());
    };

    const auto main_layer = payload(1600);
    register_payload(0, main_layer);
    std::vector<MoEOverlayPreparedExpertPayload> main_layer_table(2);
    main_layer_table[0] = main_layer;
    std::string error;
    ASSERT_TRUE(residency.registerInitialLayer(
        0,
        0,
        {true, false},
        main_layer_table,
        &error))
        << error;
    ASSERT_FALSE(residency.allInitialBanksReady());

    /*
     * The missing retained sidecar must fail precisely; finalization may not
     * fabricate an engine or silently shrink the physical graph family.
     */
    EXPECT_FALSE(residency.finalizeInitialBanksFromPreparedRegistry(
        prepared_engines,
        &error));
    EXPECT_NE(error.find("participant 0 layer 1"), std::string::npos)
        << error;
    EXPECT_FALSE(residency.allInitialBanksReady());

    const auto dormant_sidecar_layer = payload(1700);
    register_payload(1, dormant_sidecar_layer);
    ASSERT_TRUE(residency.finalizeInitialBanksFromPreparedRegistry(
        prepared_engines,
        &error))
        << error;
    EXPECT_TRUE(residency.allInitialBanksReady());
    const auto selections = residency.initialBankExpertSelections();
    ASSERT_EQ(selections.size(), 2u);
    EXPECT_EQ(selections[0].layer_idx, 0);
    EXPECT_EQ(selections[0].expert_ids, (std::vector<int>{0}));
    EXPECT_TRUE(selections[0].matches_frozen_owner_map);
    EXPECT_EQ(selections[1].layer_idx, 1);
    EXPECT_EQ(selections[1].expert_ids, (std::vector<int>{0}));
    EXPECT_TRUE(selections[1].matches_frozen_owner_map);

    /* A cached graph family observes the same completed one-way transition. */
    EXPECT_TRUE(residency.finalizeInitialBanksFromPreparedRegistry(
        prepared_engines,
        &error))
        << error;
}

TEST(Test__MoEOverlayPreparedExpertArrival,
     PublishesOneExactCompletePayloadAndRejectsIdentityDrift)
{
    MoEOverlayPreparedExpertArrival arrival;
    const auto engines = payload(3000);
    std::string error;
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Gate, engines.gate(), &error))
        << error;
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Gate, engines.gate(), &error))
        << error;
    EXPECT_FALSE(arrival.publish(
        ExpertTierWeightProjection::Gate, payload(3100).gate(), &error));
    EXPECT_FALSE(error.empty());
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Up, engines.up(), &error))
        << error;

    MoEOverlayPreparedExpertPayload resolved;
    EXPECT_FALSE(arrival.completePayload(resolved, &error));
    EXPECT_TRUE(resolved.empty());
    EXPECT_TRUE(arrival.publish(
        ExpertTierWeightProjection::Down, engines.down(), &error))
        << error;
    ASSERT_TRUE(arrival.completePayload(resolved, &error)) << error;
    EXPECT_TRUE(resolved.sameIdentity(engines));

    EXPECT_TRUE(arrival.fail("late physical failure"));
    EXPECT_FALSE(arrival.completePayload(resolved, &error));
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

    const auto preparing = fixture.authority->advanceBackground();
    ASSERT_EQ(
        preparing.status, MoEOverlayResidencyApplyStatus::Preparing)
        << preparing.error;
    /* Candidate banks are complete before, but never visible to, epoch one. */
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(2), nullptr);
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(2), nullptr);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

    const auto publishing = fixture.authority->advanceBackground();
    ASSERT_EQ(
        publishing.status, MoEOverlayResidencyApplyStatus::Publishing)
        << publishing.error;
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

    const auto published = fixture.authority->advanceBackground();
    ASSERT_EQ(published.status, MoEOverlayResidencyApplyStatus::Published)
        << published.error;
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
     GpuCandidateWithoutDevicePublisherFailsBeforeEpochPublication)
{
    auto fixture = participantMigrationFixture(/*cuda_continuation=*/true);
    auto provider = std::make_shared<IdentityTransferProvider>();
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = fixture.registry,
        .transfer_provider = provider,
        .perf_device = "cuda/cpu",
    });
    MoEOverlayTierMigrationTransport transport({
        .factory = &factory,
        .projections_per_expert = kMoEOverlayExpertProjectionCount,
        .perf_device = "cuda/cpu",
    });

    ASSERT_EQ(
        fixture.authority
            ->beginApply(fixture.transaction, transport)
            .status,
        MoEOverlayResidencyApplyStatus::Started);
    const auto failed = fixture.authority->advanceBackground();
    EXPECT_EQ(
        failed.status,
        MoEOverlayResidencyApplyStatus::PreparationFailed);
    EXPECT_NE(failed.error.find("device-bank publisher"), std::string::npos);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
    EXPECT_EQ(fixture.registry->endpoint(0)->acquire(2), nullptr);
    EXPECT_EQ(fixture.registry->endpoint(1)->acquire(2), nullptr);
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(1), nullptr);
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(1), nullptr);
}

TEST(Test__MoEOverlayParticipantMigration,
     GpuPreparationAndPublicationGateThePublicEpochAndOldBankRetirement)
{
    auto fixture = participantMigrationFixture(/*cuda_continuation=*/true);
    auto old_ticket = fixture.authority->tryAcquireTicketSnapshot();
    ASSERT_TRUE(old_ticket.has_value());
    auto provider = std::make_shared<IdentityTransferProvider>();
    auto script = std::make_shared<ScriptedDeviceBankPublication>();
    auto publisher = std::make_shared<ScriptedDeviceBankPublisher>(script);
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = fixture.registry,
        .transfer_provider = provider,
        .device_bank_publisher = publisher,
        .perf_device = "cuda/cpu",
    });
    MoEOverlayTierMigrationTransport transport({
        .factory = &factory,
        .projections_per_expert = kMoEOverlayExpertProjectionCount,
        .perf_device = "cuda/cpu",
    });

    ASSERT_EQ(
        fixture.authority
            ->beginApply(fixture.transaction, transport)
            .status,
        MoEOverlayResidencyApplyStatus::Started);

    /* Host candidates become exact-addressable while device preparation is
     * pending, but ordinary admission remains pinned to public epoch one. */
    const auto preparing = fixture.authority->advanceBackground();
    ASSERT_EQ(
        preparing.status, MoEOverlayResidencyApplyStatus::Preparing)
        << preparing.error;
    EXPECT_EQ(script->create_calls, 1);
    EXPECT_EQ(script->begin_prepare_calls, 1);
    EXPECT_EQ(script->previous_epoch, 1u);
    EXPECT_EQ(script->candidate_epoch, 2u);
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(2), nullptr);
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(2), nullptr);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
    EXPECT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Preparing);
    EXPECT_GT(script->poll_prepare_calls, 0);

    script->prepare_progress = MoEOverlayResidencyWaveProgress::Ready;
    const auto publishing = fixture.authority->advanceBackground();
    ASSERT_EQ(
        publishing.status, MoEOverlayResidencyApplyStatus::Publishing)
        << publishing.error;
    EXPECT_EQ(script->begin_publication_calls, 1);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
    auto candidate_ticket = fixture.authority->tryAcquireTicketSnapshot(2u);
    ASSERT_TRUE(candidate_ticket.has_value());
    EXPECT_EQ((*candidate_ticket)->epoch, 2u);
    candidate_ticket.reset();

    EXPECT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Publishing);
    EXPECT_GT(script->poll_publication_calls, 0);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);

    script->publication_progress = MoEOverlayResidencyWaveProgress::Ready;
    const auto published = fixture.authority->advanceBackground();
    ASSERT_EQ(
        published.status, MoEOverlayResidencyApplyStatus::Published)
        << published.error;
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 2u);
    EXPECT_EQ(script->retire_calls, 0);
    EXPECT_GT(script->poll_retirement_calls, 0)
        << "The device grace-period event must progress concurrently with a live host reader";
    const int retirement_polls_with_live_reader =
        script->poll_retirement_calls;

    old_ticket.reset();
    EXPECT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Reclaiming);
    EXPECT_GT(
        script->poll_retirement_calls,
        retirement_polls_with_live_reader);
    EXPECT_EQ(script->retire_calls, 0);
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(1), nullptr);

    script->retirement_progress = MoEOverlayResidencyWaveProgress::Ready;
    EXPECT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Idle);
    EXPECT_EQ(script->retire_calls, 1);
    EXPECT_EQ(
        script->phase, ScriptedDeviceBankPublication::Phase::Retired);
    EXPECT_EQ(fixture.registry->endpoint(0)->acquire(1), nullptr);
    EXPECT_EQ(fixture.registry->endpoint(1)->acquire(1), nullptr);
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(2), nullptr);
}

TEST(Test__MoEOverlayParticipantMigration,
     FailedGpuPreparationDrainsAbortBeforeReleasingWaveOwnership)
{
    auto fixture = participantMigrationFixture(/*cuda_continuation=*/true);
    auto provider = std::make_shared<IdentityTransferProvider>();
    auto script = std::make_shared<ScriptedDeviceBankPublication>();
    auto publisher = std::make_shared<ScriptedDeviceBankPublisher>(script);
    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = fixture.registry,
        .transfer_provider = provider,
        .device_bank_publisher = publisher,
        .perf_device = "cuda/cpu",
    });
    MoEOverlayTierMigrationTransport transport({
        .factory = &factory,
        .projections_per_expert = kMoEOverlayExpertProjectionCount,
        .perf_device = "cuda/cpu",
    });

    ASSERT_EQ(
        fixture.authority
            ->beginApply(fixture.transaction, transport)
            .status,
        MoEOverlayResidencyApplyStatus::Started);
    ASSERT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Preparing);
    ASSERT_NE(fixture.registry->endpoint(0)->acquire(2), nullptr);

    script->prepare_progress = MoEOverlayResidencyWaveProgress::Failed;
    const auto failed = fixture.authority->advanceBackground();
    EXPECT_EQ(
        failed.status,
        MoEOverlayResidencyApplyStatus::PreparationFailed);
    EXPECT_NE(failed.error.find("scripted device preparation failed"),
              std::string::npos);
    EXPECT_EQ(script->abort_calls, 1);
    EXPECT_EQ(fixture.authority->snapshot()->epoch, 1u);
    EXPECT_EQ(fixture.registry->endpoint(0)->acquire(2), nullptr);
    EXPECT_EQ(fixture.registry->endpoint(1)->acquire(2), nullptr);

    /* The wave remains owned by the authority until the exact asynchronous
     * device abort edge reaches Ready. */
    EXPECT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Reclaiming);
    EXPECT_GT(script->poll_abort_calls, 0);
    EXPECT_NE(
        script->phase, ScriptedDeviceBankPublication::Phase::Aborted);

    script->abort_progress = MoEOverlayResidencyWaveProgress::Ready;
    EXPECT_EQ(
        fixture.authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Idle);
    EXPECT_EQ(
        script->phase, ScriptedDeviceBankPublication::Phase::Aborted);
    EXPECT_NE(fixture.registry->endpoint(0)->acquire(1), nullptr);
    EXPECT_NE(fixture.registry->endpoint(1)->acquire(1), nullptr);
}

TEST(Test__MoEOverlayParticipantMigration,
     IncompleteArrivalFailsPreparationAndNeverExposesCandidateEpoch)
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
    EXPECT_EQ(
        failed.status,
        MoEOverlayResidencyApplyStatus::PreparationFailed);
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
        prepareAndInstall(*endpoint, unrelated_candidate, &error),
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
    external.layers[0].setResidentExpert(0, payload(4000));
    std::string error;
    ASSERT_EQ(
        prepareAndInstall(*p1, external, &error),
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
    EXPECT_FALSE(prepared.inactive_bank->beginPrepare(&error));
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
    ASSERT_TRUE(prepared.inactive_bank->beginPrepare(nullptr));
    EXPECT_EQ(
        prepared.inactive_bank->pollPrepare(nullptr),
        MoEOverlayResidencyWaveProgress::Ready);
    ASSERT_TRUE(prepared.inactive_bank->beginPublication(nullptr));
    EXPECT_EQ(
        prepared.inactive_bank->pollPublication(nullptr),
        MoEOverlayResidencyWaveProgress::Ready);
    EXPECT_EQ(
        prepared.inactive_bank->pollRetirementFence(nullptr),
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

TEST(Test__MoEOverlayParticipantResidency,
     DeviceServiceSnapshotsReplaceOnlyMonotonicCumulativeEvidence)
{
    MoEOverlayParticipantResidency endpoint({
        .participant_id = 3,
        .device = DeviceId::cuda(0),
        .num_layers = 2,
        .num_experts = 4,
        .collect_economy_service_measurements = true,
    });
    std::vector<MoEOverlayParticipantLayerServiceTotals> imported(2);
    for (int layer = 0; layer < 2; ++layer)
    {
        imported[static_cast<std::size_t>(layer)].participant_id = 3;
        imported[static_cast<std::size_t>(layer)].layer = layer;
        for (std::size_t phase = 0;
             phase < kExpertHistogramProductionSourceCount;
             ++phase)
        {
            imported[static_cast<std::size_t>(layer)]
                .total_nanoseconds[phase] = 100u + layer * 10u + phase;
            imported[static_cast<std::size_t>(layer)]
                .activation_count[phase] = 10u + phase;
            imported[static_cast<std::size_t>(layer)]
                .sample_count[phase] = 1u;
        }
    }

    std::string error;
    ASSERT_TRUE(endpoint.importServiceMeasurements(imported, &error))
        << error;
    EXPECT_TRUE(endpoint.importServiceMeasurements(imported, &error))
        << error;
    EXPECT_EQ(
        endpoint.recordServiceMeasurement(
            0, ExpertHistogramSource::DecodeToken, 1u, 1u),
        MoEOverlayServiceMeasurementRecordStatus::Invalid);

    auto newer = imported;
    for (auto &row : newer)
    {
        for (std::size_t phase = 0;
             phase < kExpertHistogramProductionSourceCount;
             ++phase)
        {
            row.total_nanoseconds[phase] += 50u;
            row.activation_count[phase] += 5u;
            row.sample_count[phase] += 1u;
        }
    }
    ASSERT_TRUE(endpoint.importServiceMeasurements(newer, &error))
        << error;

    std::vector<MoEOverlayParticipantLayerServiceTotals> observed;
    ASSERT_TRUE(endpoint.trySnapshotServiceMeasurements(&observed));
    EXPECT_EQ(observed.size(), newer.size());
    for (std::size_t layer = 0; layer < newer.size(); ++layer)
    {
        EXPECT_EQ(
            observed[layer].total_nanoseconds,
            newer[layer].total_nanoseconds);
        EXPECT_EQ(
            observed[layer].activation_count,
            newer[layer].activation_count);
        EXPECT_EQ(observed[layer].sample_count, newer[layer].sample_count);
    }

    auto regressed = newer;
    --regressed[1].sample_count[2];
    EXPECT_FALSE(endpoint.importServiceMeasurements(regressed, &error));
    EXPECT_NE(error.find("conflicts"), std::string::npos);
}
