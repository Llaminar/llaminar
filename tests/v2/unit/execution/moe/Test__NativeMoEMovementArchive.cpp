/**
 * @file Test__NativeMoEMovementArchive.cpp
 * @brief Device-free adversarial tests of terminal native movement evidence.
 *
 * Real publisher/copy behavior belongs in CUDA/ROCm preflight. Here the same
 * POD receipts prove request-reset identity, immutable history, bounded loss,
 * exact payload accounting, and atomic rejection without loading a model or
 * enabling PerfStats. Both GPU device kinds share this public contract.
 */
#include "execution/moe/NativeMoEMovementArchive.h"
#include "execution/moe/NativeMoEMovementRequestIdentity.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/runner/OrchestrationRunner.h"
#include "app/modes/MoEMovementLedgerJson.h"
#include "../../../mocks/MockModelContext.h"
#include "../../../mocks/MockLocalTPContext.h"
#include <gtest/gtest.h>
#include <array>
#include <stdexcept>

namespace llaminar2::test
{
    namespace
    {
        /** @return One legal domain; root/rank/priority are deliberately not zero. */
        NativeMoEMovementArchiveConfig geometry(bool rocm = false)
        {
            return {.workspace_generation = 7, .layers = 2, .experts = 8,
                .wave_capacity = 2, .edge_capacity = 4,
                .participants = {{rocm ? DeviceId::rocm(2) : DeviceId::cuda(1), 3, 17},
                                 {rocm ? DeviceId::rocm(0) : DeviceId::cuda(0), 3, 17}}};
        }

        /** Own an actual journal view; construct receipts through its shared publisher. */
        struct Snapshot
        {
            DeviceMoERebalanceMovementJournalState state;
            std::array<DeviceMoERebalanceMovementWave, 2> waves{};
            std::array<DeviceMoERebalanceMovementEdge, 4> edges{};

            /** @brief Commit one complete reciprocal pair using the device protocol. */
            void commit(std::uint64_t epoch, std::uint32_t command_epoch = 1)
            {
                const DeviceMoERebalanceLoadSpreadProof proof{
                    .accepted_spread_improvement = 40, .pre_wave_spread = 100, .post_wave_spread = 60,
                    .pre_wave_total = 200, .post_wave_total = 200,
                    .pre_participant_spread = 0, .post_participant_spread = 20,
                    .pre_participant_total = 200, .post_participant_total = 200,
                    .requested_payload_slots = 1, .minimum_improvement_per_slot = 40,
                    .maximum_post_spread_per_mille = 300, .ownership_swap_accepts = 1};
                DeviceMoERebalanceMovementJournalView view{&state, waves.data(), edges.data(), 2, 4};
                auto append = prepareDeviceMoEMovementJournalAppend(view, epoch, command_epoch, 2, proof, 6000);
                ASSERT_NE(append.disposition, DeviceMoEMovementJournalDisposition::Invalid);
                if (append.disposition == DeviceMoEMovementJournalDisposition::Record)
                {
                    edges[append.first_edge] = {0, 3, 0, 1, 90, 8192};
                    edges[append.first_edge + 1] = {0, 4, 1, 0, 10, 8192};
                }
                ASSERT_TRUE(commitDeviceMoEMovementJournalAppend(view, append));
            }

            /** @return First newly archived edge, using exactly the populated ranges. */
            std::size_t observe(NativeMoEMovementArchive &archive, std::uint64_t session = 1,
                std::uint64_t generation = 7) const
            {
                return archive.observe(session, generation, state,
                    std::span(waves).first(state.committed_waves),
                    std::span(edges).first(state.committed_edges));
            }
        };

        /** Device-free child that exposes only already-completed evidence. */
        class ReceiptRunner final : public IInferenceRunner
        {
        public:
            MoEOptimizationMovementLedger receipt;
            MoEOptimizationStatus status;
            /** @return False: this diagnostic fixture never executes inference. */
            bool forward(const int *, int) override { return false; }
            /** @return No tensor allocation is needed for terminal evidence. */
            const float *logits() const override { return nullptr; }
            /** @return Minimal model-independent vocabulary geometry. */
            int vocab_size() const override { return 1; }
            /** @brief Reset data does not discard model-lifetime receipts. */
            void clear_cache() override {}
            /** @return No token position is advanced by this fixture. */
            int get_position() const override { return 0; }
            /** @return Declarative metadata only; this is not a capture proof. */
            ExecutionPath executionPath() const override { return ExecutionPath::GRAPH; }
            /** @return Stable diagnostic architecture identity. */
            const char *architecture() const override { return "receipt-fixture"; }
            /** @copydoc IInferenceRunner::moeOptimizationStatus */
            MoEOptimizationStatus moeOptimizationStatus() const override { return status; }
            /** @copydoc IInferenceRunner::moeOptimizationMovementLedger */
            MoEOptimizationMovementLedger moeOptimizationMovementLedger() const override { return receipt; }
        };
    }

    TEST(Test__NativeMoEMovementArchive, BothBackendsPreserveNativeUnitsEndpointsAndActualBytes)
    {
        for (bool rocm : {false, true})
        {
            NativeMoEMovementArchive archive(geometry(rocm));
            Snapshot snapshot;
            snapshot.commit(9);
            EXPECT_EQ(snapshot.observe(archive), 0);
            ASSERT_EQ(archive.ledger().edges.size(), 2);
            const auto &edge = archive.ledger().edges.front();
            EXPECT_TRUE(edge.valid());
            EXPECT_EQ(edge.source_priority, 17);
            EXPECT_EQ(edge.source_world_rank, 3);
            EXPECT_EQ(edge.source_device, geometry(rocm).participants[0].device);
            EXPECT_EQ(edge.transaction, 9);
            EXPECT_EQ(edge.axis, MoEOptimizationMovementAxis::ParticipantPlacement);
            EXPECT_EQ(edge.activation_count, 90);
            EXPECT_EQ(archive.totals().physical_bytes, 6000); // Not 2 * padded 8192-byte descriptors.
            EXPECT_EQ(archive.totals().same_priority_moves, 2);
            ASSERT_EQ(archive.ledger().economy.size(), 1);
            EXPECT_TRUE(archive.ledger().economy.front().valid());
            EXPECT_TRUE(std::holds_alternative<DeviceMoERebalanceLoadSpreadProof>(archive.ledger().economy.front().proof));
        }
    }

    TEST(Test__NativeMoEMovementArchive, RepeatedObservationIsIdempotentAndAppendOnly)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot snapshot;
        snapshot.commit(9);
        snapshot.observe(archive);
        const auto original = archive.ledger().edges;
        EXPECT_EQ(snapshot.observe(archive), 2);
        EXPECT_EQ(archive.totals().physical_bytes, 6000);
        snapshot.commit(10, 2);
        EXPECT_EQ(snapshot.observe(archive), 2);
        EXPECT_TRUE(std::equal(original.begin(), original.end(), archive.ledger().edges.begin()));
        EXPECT_EQ(archive.totals().physical_bytes, 12000);
    }

    TEST(Test__NativeMoEMovementArchive, RequestResetRetainsHistoryAndCannotReuseModelEpoch)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot first;
        first.commit(9);
        first.observe(archive);
        Snapshot next;
        next.observe(archive, 2); // Full-prefix hits can have no new movement.
        next.commit(9);
        EXPECT_THROW(next.observe(archive, 2), std::invalid_argument);
        next = {};
        next.commit(10); // Command epoch restarts at one, model epoch does not.
        EXPECT_NO_THROW(next.observe(archive, 2));
        ASSERT_EQ(archive.ledger().economy.size(), 2);
        EXPECT_EQ(archive.ledger().economy.back().transaction, 10);
        EXPECT_THROW(first.observe(archive), std::invalid_argument);
    }

    TEST(Test__NativeMoEMovementArchive, EveryChangedPublishedFieldRejectsAtomically)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot snapshot;
        snapshot.commit(9);
        snapshot.observe(archive);
        for (int field = 0; field != 9; ++field)
        {
            auto changed = snapshot;
            switch (field)
            {
            case 0: ++changed.waves[0].physical_payload_bytes; break;
            case 1: ++changed.waves[0].command_epoch; break;
            case 2: ++changed.edges[0].activation_count; break;
            case 3: ++changed.edges[0].estimated_weight_bytes; break;
            case 4: changed.edges[0].expert = 2; break;
            case 5: ++changed.waves[0].proof.pre_wave_total; break;
            case 6: changed = {}; break;
            case 7: changed.edges[1].destination_participant = 1; break;
            case 8: ++changed.waves[0].candidate_epoch; ++changed.state.last_candidate_epoch; break;
            }
            EXPECT_THROW(changed.observe(archive), std::invalid_argument) << field;
            EXPECT_EQ(archive.ledger().edges.size(), 2);
            EXPECT_EQ(archive.totals().physical_bytes, 6000);
        }
        EXPECT_NO_THROW(snapshot.observe(archive));
    }

    TEST(Test__NativeMoEMovementArchive, InvalidFreshSnapshotCannotPublishAnyPartialHistory)
    {
        for (int field = 0; field != 6; ++field)
        {
            NativeMoEMovementArchive archive(geometry());
            Snapshot snapshot;
            snapshot.commit(9);
            snapshot.commit(10, 2);
            switch (field)
            {
            case 0: snapshot.edges[3].source_participant = 8; break;
            case 1: snapshot.waves[1].first_edge = 0; break;
            case 2: snapshot.waves[1].physical_payload_bytes = 0; break;
            case 3: snapshot.waves[1].proof.accepted_spread_improvement = 0; break;
            case 4: snapshot.state.last_candidate_epoch = 11; break;
            case 5: snapshot.edges[2].estimated_weight_bytes = 0; break;
            }
            EXPECT_THROW(snapshot.observe(archive), std::invalid_argument) << field;
            EXPECT_TRUE(archive.ledger().edges.empty());
            EXPECT_EQ(archive.totals().physical_bytes, 0);
        }
    }

    TEST(Test__NativeMoEMovementArchive, BoundedArchiveNeverOverwritesAndLossIsStickyAcrossReset)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot snapshot;
        snapshot.commit(9);
        snapshot.commit(10, 2);
        snapshot.observe(archive);
        const auto original = archive.ledger().edges;
        snapshot.commit(11, 3); // Device journal exhaustion.
        snapshot.observe(archive);
        EXPECT_FALSE(archive.ledger().complete());
        EXPECT_EQ(archive.ledger().discarded_edges, 2);
        snapshot.observe(archive);
        EXPECT_EQ(archive.ledger().discarded_edges, 2);
        Snapshot next;
        next.commit(12); // Device space is fresh; model archive remains full.
        next.observe(archive, 2);
        EXPECT_EQ(archive.ledger().discarded_edges, 4);
        EXPECT_EQ(archive.ledger().discarded_economy_records, 2);
        EXPECT_EQ(archive.ledger().edges, original);
        EXPECT_EQ(archive.totals().transactions, 3); // Only actual retained device receipts count.
    }

    TEST(Test__NativeMoEMovementArchive, ArenaRebindingAndInvalidDomainAreRejected)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot snapshot;
        snapshot.commit(9);
        EXPECT_THROW(snapshot.observe(archive, 1, 8), std::invalid_argument);
        EXPECT_THROW(snapshot.observe(archive, 0), std::invalid_argument);
        for (int field = 0; field != 5; ++field)
        {
            auto invalid = geometry();
            switch (field)
            {
            case 0: invalid.participants[1].device = DeviceId::cpu(); break;
            case 1: invalid.participants[1].priority = 18; break;
            case 2: invalid.participants[1].world_rank = -1; break;
            case 3: invalid.participants[1] = invalid.participants[0]; break;
            case 4: invalid.edge_capacity = 1; break;
            }
            EXPECT_THROW(NativeMoEMovementArchive{invalid}, std::invalid_argument) << field;
        }
    }

    TEST(Test__NativeMoEMovementArchive, LossCountersRequireNewPublicationAndTotalsCannotOverflow)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot snapshot;
        snapshot.commit(9);
        snapshot.commit(10, 2);
        snapshot.commit(11, 3);
        snapshot.observe(archive);
        auto forged = snapshot;
        ++forged.state.discarded_waves;
        forged.state.discarded_edges += 2;
        EXPECT_THROW(forged.observe(archive), std::invalid_argument);
        forged = snapshot;
        ++forged.state.last_candidate_epoch;
        EXPECT_THROW(forged.observe(archive), std::invalid_argument);

        NativeMoEMovementArchive overflow(geometry());
        Snapshot large;
        large.commit(9);
        large.waves[0].physical_payload_bytes = UINT64_MAX;
        large.observe(overflow);
        large.commit(10, 2);
        EXPECT_THROW(large.observe(overflow), std::invalid_argument);
        EXPECT_EQ(overflow.totals().physical_bytes, UINT64_MAX);
        EXPECT_EQ(overflow.ledger().edges.size(), 2);
    }

    TEST(Test__NativeMoEMovementArchive, PublicRunnerForwardsNonFirstRootAndRetainsWireProof)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot snapshot;
        snapshot.commit(9);
        snapshot.observe(archive);
        auto follower = std::make_unique<ReceiptRunner>();
        auto root = std::make_unique<ReceiptRunner>();
        auto *follower_ptr = follower.get();
        root->receipt = archive.ledger();
        root->status = {.authority = MoEOptimizationAuthority::Device,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::DeviceOwned,
            .published_movement_waves = archive.totals().transactions, .completed_movement = archive.totals()};
        std::vector<std::unique_ptr<IInferenceRunner>> children;
        children.push_back(std::move(follower));
        children.push_back(std::move(root));
        RankOrchestrator::Config rank_config;
        rank_config.mode = RankOrchestrator::ParallelismMode::TP;
        rank_config.devices = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
        auto collective = std::make_unique<MockLocalTPContext>();
        collective->setDevices(rank_config.devices);
        auto rank = RankOrchestrator::createForTest(MockModelContext::createMinimal(),
            std::move(children), std::move(collective), rank_config);
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
        config.prefix_cache.enabled = false;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        RankExecutionPlan plan;
        plan.primary_device = *config.device_for_this_rank;
        plan.runtime.prefix_cache.enabled = false;
        plan.runtime.prefix_cache.storage_mode = PrefixCacheStorageMode::Disabled;
        OrchestrationRunner runner(config, plan, std::move(rank));
        EXPECT_EQ(runner.moeOptimizationStatus().authority, MoEOptimizationAuthority::Device);
        EXPECT_EQ(runner.moeOptimizationStatus().completed_movement.physical_bytes, 6000);
        EXPECT_FALSE(runner.moeOptimizationStatus().quiescentBetweenWaves());
        EXPECT_EQ(runner.moeOptimizationMovementLedger().edges, archive.ledger().edges);
        const auto json = moeMovementLedgerJson(runner.moeOptimizationMovementLedger());
        EXPECT_EQ(json.at("schema"), 2);
        EXPECT_EQ(json.at("economy").at(0).at("policy"), "native_load_spread");
        EXPECT_EQ(json.at("edges").at(0).at("transaction"), 9);
        // Two roots must not silently select the first child or add duplicates.
        follower_ptr->status.authority = MoEOptimizationAuthority::Device;
        EXPECT_THROW(runner.moeOptimizationStatus(), std::logic_error);
        EXPECT_THROW(runner.moeOptimizationMovementLedger(), std::logic_error);
        follower_ptr->status = {};
    }

    TEST(Test__NativeMoEMovementArchive, RequestBindingDoesNotRequireAHipSchedulerAction)
    {
        // These are exactly the common initialization fields. CUDA never
        // needs to publish a host scheduler decision to certify movement.
        DeviceMoERebalanceDispatchTicket ticket;
        ticket.magic = DeviceMoERebalanceDispatchTicket::kMagic;
        ticket.abi_version = DeviceMoERebalanceDispatchTicket::kABIVersion;
        ticket.session_epoch_low = 3;
        ticket.workspace_generation_low = 7;
        ticket.participant_id = 1;
        ticket.participant_count = 2;
        EXPECT_FALSE(ticket.matchesLifecycle(3, 7, 1, 2));
        EXPECT_EQ(nativeMoEMovementArchiveRequestEpoch(ticket, 7, 1, 2), 3);
        for (int field = 0; field != 6; ++field)
        {
            auto wrong = ticket;
            switch (field)
            {
            case 0: wrong.magic = 0; break;
            case 1: ++wrong.abi_version; break;
            case 2: wrong.session_epoch_low = 0; break;
            case 3: ++wrong.workspace_generation_low; break;
            case 4: wrong.participant_id = 0; break;
            case 5: ++wrong.participant_count; break;
            }
            EXPECT_THROW((void)nativeMoEMovementArchiveRequestEpoch(wrong, 7, 1, 2), std::invalid_argument) << field;
        }
        // Publishing a valid HIP scheduler decision preserves that identity.
        ticket.controller_version = moe_rebalance_abi::kVersion;
        ticket.healthy = 1;
        ticket.decode_rounds_until_maintenance = 4;
        EXPECT_TRUE(ticket.matchesLifecycle(3, 7, 1, 2));
        EXPECT_EQ(nativeMoEMovementArchiveRequestEpoch(ticket, 7, 1, 2), 3);
    }
}
