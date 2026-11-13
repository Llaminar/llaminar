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
#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/DeviceMoERebalanceWorkspaceContract.h"
#include "execution/moe/MoEOverlayDeviceEpochProtocol.h"
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
            EXPECT_EQ(edge.source.whole().world_rank.value_or(-1), 3);
            EXPECT_EQ(edge.source.whole().device, geometry(rocm).participants[0].device);
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
        EXPECT_EQ(json.at("schema"), 4);
        EXPECT_EQ(json.at("scope"), "model_lifetime");
        EXPECT_EQ(json.at("edges").size(), 2);
        // Forward the complete typed identity, including the non-first root's
        // original rank/device. Estimated edge bytes remain distinct from the
        // actual6000-byte native completion receipt asserted above.
        const auto &wire_edge = json.at("edges").at(0);
        EXPECT_EQ(wire_edge.at("source"), (nlohmann::json{{"kind", "whole"},
            {"participant", 0}, {"device", "CUDA:1"}, {"world_rank", 3}}));
        EXPECT_EQ(wire_edge.at("destination"), (nlohmann::json{{"kind", "whole"},
            {"participant", 1}, {"device", "CUDA:0"}, {"world_rank", 3}}));
        EXPECT_EQ(wire_edge.at("estimated_weight_bytes"), 8192);
        EXPECT_FALSE(wire_edge.contains("source_participant"));
        EXPECT_FALSE(wire_edge.contains("destination_participant"));
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
    /** @test Actual shared device publication body counts no-ops, never skipped/failed work. */
    TEST(Test__NativeMoEMovementArchive, CompletedDecisionPublisherReplay)
    {
        static_assert(sizeof(DeviceMoERebalanceGraphControllerState) == moe_rebalance_abi::kGraphControllerBytes);
        DeviceMoERebalanceGraphControllerState controller;
        controller.participant_count = 3; controller.participant_id = 2;
        DeviceMoERebalanceStatus policy;
        policy.windows_observed = 1; policy.window_required_slots = 4; policy.window_ready_slots = 4;
        policy.pre_wave_load_total = 123; policy.dynamic_ownership_swap_rejections = 2;
        for (std::uint32_t replay = 0; replay < 20; ++replay)
        {
            controller.decode_rounds_committed += 3;
            const auto before = controller.completed_decision;
            auto failed = policy; failed.status_code = static_cast<uint32_t>(DeviceMoERebalanceStatusCode::InvalidRuntime);
            auto append = prepareDeviceMoECompletedDecision(controller, failed, 2, 0, 0);
            EXPECT_EQ(append.disposition, DeviceMoECompletedDecisionDisposition::Invalid);
            EXPECT_FALSE(commitDeviceMoECompletedDecision(controller.completed_decision, append));
            EXPECT_EQ(controller.completed_decision, before);
            auto zero_work = policy; zero_work.status_code = 1; zero_work.windows_observed = 0;
            zero_work.window_ready_slots = 0; zero_work.skipped_not_ready = 1;
            append = prepareDeviceMoECompletedDecision(controller, zero_work, 2, 0, 0);
            ASSERT_EQ(append.disposition, DeviceMoECompletedDecisionDisposition::Inactive);
            ASSERT_TRUE(commitDeviceMoECompletedDecision(controller.completed_decision, append));
            EXPECT_EQ(controller.completed_decision, before);
            append = prepareDeviceMoECompletedDecision(controller, policy, 2, 0, 0);
            ASSERT_EQ(append.disposition, DeviceMoECompletedDecisionDisposition::Record);
            EXPECT_EQ(controller.completed_decision, before); // Preparation is not completion.
            ASSERT_TRUE(commitDeviceMoECompletedDecision(controller.completed_decision, append));
            EXPECT_FALSE(commitDeviceMoECompletedDecision(controller.completed_decision, append));
            EXPECT_EQ(controller.completed_decision.completed_windows, replay + 1);
            EXPECT_EQ(controller.completed_decision.candidate_epoch, 2); // Valid no-op epochs repeat.
            EXPECT_EQ(controller.completed_decision.routed_observations, 123);
        }
        auto overflow = controller; overflow.completed_decision.completed_windows = UINT64_MAX;
        EXPECT_EQ(prepareDeviceMoECompletedDecision(overflow, policy, 2, 0, 0).disposition,
            DeviceMoECompletedDecisionDisposition::Invalid);
        auto malformed = policy; malformed.window_ready_slots = 3;
        EXPECT_EQ(prepareDeviceMoECompletedDecision(controller, malformed, 2, 0, 0).disposition,
            DeviceMoECompletedDecisionDisposition::Invalid);
        malformed = policy; malformed.version--;
        EXPECT_EQ(prepareDeviceMoECompletedDecision(controller, malformed, 2, 0, 0).disposition,
            DeviceMoECompletedDecisionDisposition::Invalid);
        auto singleton = controller; singleton.participant_count = 1; singleton.participant_id = 0;
        EXPECT_EQ(prepareDeviceMoECompletedDecision(singleton, policy, 2, 0, 0).disposition,
            DeviceMoECompletedDecisionDisposition::Invalid);
        // Request reset's native initializer assigns a complete fresh controller.
        controller = {}; controller.participant_count = 3;
        policy.dynamic_ownership_swap_accepts = 1; policy.last_epoch = 1;
        const auto moving = prepareDeviceMoECompletedDecision(controller, policy, 3, 1, 2);
        ASSERT_TRUE(commitDeviceMoECompletedDecision(controller.completed_decision, moving));
        EXPECT_EQ(controller.completed_decision.completed_windows, 1);
        EXPECT_EQ(controller.completed_decision.command_epoch, 1);
        EXPECT_EQ(controller.completed_decision.command_count, 2);
        EXPECT_EQ(controller.completed_decision.accepted_cycles, 1);
    }

    /** @test Immutable public decision evidence retains request identity and cumulative progress. */
    TEST(Test__NativeMoEMovementArchive, CompletedDecisionResetAndRepeatedTerminalObservation)
    {
        NativeMoEMovementArchive archive(geometry());
        Snapshot empty;
        DeviceMoERebalanceCompletedDecision receipt{3, 2, 123, 0, 0, 0, 2, 9, 1};
        const auto observe = [&](std::uint64_t request) {
            return archive.observe(request, 7, empty.state, {}, {}, receipt);
        };
        ASSERT_NO_THROW(observe(41));
        ASSERT_TRUE(archive.lastDecision());
        EXPECT_EQ(archive.completedDecisionWindows(), 3);
        EXPECT_TRUE(archive.lastDecision()->noMovement());
        EXPECT_EQ(archive.lastDecision()->transaction, 3);
        EXPECT_EQ(archive.lastDecision()->source_request_epoch, 41);
        EXPECT_EQ(archive.lastDecision()->snapshot_observations, 123);
        MoEOptimizationStatus status{.authority = MoEOptimizationAuthority::Device,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::DeviceOwned,
            .completed_decision_windows = archive.completedDecisionWindows(), .last_decision = archive.lastDecision()};
        const auto stamp = status.progressStamp();
        EXPECT_EQ(moeOptimizationStatusJson(status)["last_decision"]["source_request_epoch"], 41);
        EXPECT_NO_THROW(observe(41)); EXPECT_EQ(archive.completedDecisionWindows(), 3);
        EXPECT_EQ(status.progressStamp().relationTo(stamp), MoEOptimizationProgressRelation::Unchanged);
        ++receipt.completed_windows; receipt.committed_decode_rounds = 12;
        EXPECT_NO_THROW(observe(41)); EXPECT_EQ(archive.completedDecisionWindows(), 4);
        status.completed_decision_windows = archive.completedDecisionWindows(); status.last_decision = archive.lastDecision();
        EXPECT_EQ(status.progressStamp().relationTo(stamp), MoEOptimizationProgressRelation::Advanced);
        // A no-work new request preserves the prior genuine last decision.
        receipt = {};
        EXPECT_NO_THROW(observe(42)); EXPECT_EQ(archive.completedDecisionWindows(), 4);
        EXPECT_EQ(archive.lastDecision()->source_request_epoch, 41);
        receipt = {1, 2, 37, 0, 0, 0, 1, 4, 1};
        EXPECT_NO_THROW(observe(42)); EXPECT_EQ(archive.completedDecisionWindows(), 5);
        EXPECT_EQ(archive.lastDecision()->transaction, 1); // Request-local ordinal, not invented global txn.
        EXPECT_EQ(archive.lastDecision()->source_request_epoch, 42);
        EXPECT_NO_THROW(observe(42)); EXPECT_EQ(archive.completedDecisionWindows(), 5);
        // Existing non-request-scoped topology/host receipt semantics stay valid.
        MoEOptimizationDecisionReceipt legacy{.transaction = 1, .candidate_epoch = 1};
        EXPECT_TRUE(legacy.valid()); EXPECT_EQ(legacy.source_request_epoch, 0);
        EXPECT_TRUE(legacy.rejected_cycles_observed);
        legacy.rejected_cycles_observed = false; legacy.rejected_cycles = 1;
        EXPECT_FALSE(legacy.valid());
        status.last_decision = legacy;
        EXPECT_THROW(moeOptimizationStatusJson(status), std::invalid_argument);
    }

    /** @test Corrupt/stale completion evidence cannot mutate accepted public history. */
    TEST(Test__NativeMoEMovementArchive, CompletedDecisionMalformedEvidenceIsAtomic)
    {
        NativeMoEMovementArchive archive(geometry(true));
        DeviceMoERebalanceMovementJournalState journal;
        const DeviceMoERebalanceCompletedDecision original{2, 4, 123, 7, 2, 1, 0, 8, 0};
        archive.observe(1, 7, journal, {}, {}, original);
        for (int field = 0; field < 9; ++field)
        {
            auto changed = original;
            switch (field) {
            case 0: --changed.completed_windows; break;
            case 1: ++changed.routed_observations; break;
            case 2: ++changed.command_epoch; break;
            case 3: ++changed.command_count; break;
            case 4: ++changed.accepted_cycles; break;
            case 5: ++changed.rejected_cycles; break;
            case 6: --changed.committed_decode_rounds; break;
            case 7: changed.observation_flags = 2; break;
            case 8: changed.candidate_epoch = 0; break;
            }
            EXPECT_THROW(archive.observe(1, 7, journal, {}, {}, changed), std::invalid_argument) << field;
            EXPECT_EQ(archive.completedDecisionWindows(), 2);
            EXPECT_EQ(archive.lastDecision()->snapshot_observations, 123);
        }
        auto stale = original; ++stale.completed_windows; --stale.candidate_epoch;
        EXPECT_THROW(archive.observe(1, 7, journal, {}, {}, stale), std::invalid_argument);
        EXPECT_THROW(archive.observe(2, 7, journal, {}, {}, stale), std::invalid_argument);
        auto reset = original; reset.completed_windows = 0;
        EXPECT_THROW(archive.observe(1, 7, journal, {}, {}, reset), std::invalid_argument);
        EXPECT_THROW(archive.observe(1, 8, journal, {}, {}, original), std::invalid_argument);
    }

    /** @test Existing canonical workspace BOM admits the complete appended controller ABI. */
    TEST(Test__NativeMoEMovementArchive, CompletedDecisionUsesCanonicalControllerBOM)
    {
        DeviceMoERebalanceConfig config;
        config.num_layers = 2; config.num_experts = 8; config.participant_count = 3;
        config.flags |= static_cast<std::uint32_t>(DeviceMoERebalanceFlags::CompletedDynamicDecisions);
        const auto capacity = DeviceMoERebalanceWorkspaceCapacity::fromRuntime(config, 0, 0,
            DeviceMoERebalanceStagePhase::PlanCopyApply, DeviceMoERebalanceTransferMode::ResidentOnly);
        EXPECT_TRUE(hasDeviceMoERebalanceFlag(capacity.flags, DeviceMoERebalanceFlags::CompletedDynamicDecisions));
        const auto requirements = DeviceMoERebalanceWorkspaceContract::requirements({capacity, 0, "decision_bom"});
        std::size_t controllers = 0;
        for (const auto &buffer : requirements.buffers)
            if (buffer.name == std::string(DeviceMoERebalanceWorkspaceContract::WS_CONTROLLER_STATE) + "_decision_bom") {
                ++controllers;
                EXPECT_EQ(buffer.size_bytes, sizeof(DeviceMoERebalanceGraphControllerState));
                EXPECT_EQ(buffer.size_bytes, 408);
                EXPECT_EQ(buffer.alignment, 256);
            }
        EXPECT_EQ(controllers, 1);
    }

    /** @test Busy reserve retains moving proof while later scratch describes a skipped new window. */
    TEST(Test__NativeMoEMovementArchive, CompletedDecisionRetainedMovingProofSurvivesBusyReserve)
    {
        DeviceMoEOverlayEpochControl control;
        MoEOverlayDeviceEpochProtocol::initialize(control, 1);
        MoEOverlayDeviceEpochProtocol epochs(control);
        auto held = epochs.tryAcquirePublished(); ASSERT_TRUE(held);
        ASSERT_TRUE(epochs.reserveCandidate(2)); epochs.markCandidateReady(2);
        ASSERT_TRUE(epochs.publishReadyCandidate(2).valid());
        DeviceMoERebalanceGraphControllerState controller; controller.participant_count = 3;
        controller.decode_rounds_committed = 12;
        const DeviceMoERebalanceLoadSpreadProof retained{40, 100, 60, 200, 200, 0, 20, 200, 200, 1, 40, 300, 1};
        EXPECT_FALSE(epochs.reserveCandidate(3)); // Old original inference ticket still pins the bank.
        EXPECT_EQ(controller.completed_decision.completed_windows, 0);
        DeviceMoERebalanceStatus later; later.status_code = static_cast<std::uint32_t>(DeviceMoERebalanceStatusCode::WindowNotReady);
        later.skipped_not_ready = later.skipped_busy_wave = 1;
        later.pre_wave_load_total = 999; // A different planner pass has no authority over retained commands.
        EXPECT_EQ(prepareDeviceMoECompletedDecision(controller, later, 3, 7, 2).disposition,
            DeviceMoECompletedDecisionDisposition::Invalid);
        ASSERT_TRUE(epochs.release(*held)); ASSERT_TRUE(epochs.tryRetire(1));
        ASSERT_TRUE(epochs.reserveCandidate(3));
        const auto append = prepareDeviceMoECompletedOwnershipDecision(controller, retained, 3, 7, 2);
        ASSERT_EQ(append.disposition, DeviceMoECompletedDecisionDisposition::Record);
        EXPECT_EQ(controller.completed_decision.completed_windows, 0); // Preparation still is not completion.
        epochs.markCandidateReady(3); ASSERT_TRUE(epochs.publishReadyCandidate(3).valid());
        ASSERT_TRUE(commitDeviceMoECompletedDecision(controller.completed_decision, append));
        EXPECT_FALSE(commitDeviceMoECompletedDecision(controller.completed_decision, append));
        const auto &receipt = controller.completed_decision;
        EXPECT_EQ(receipt.completed_windows, 1); EXPECT_EQ(receipt.command_epoch, 7);
        EXPECT_EQ(receipt.routed_observations, 200); EXPECT_EQ(receipt.accepted_cycles, 1);
        EXPECT_EQ(receipt.observation_flags & kNativeMoEPlannerRejectionsObserved, 0); // Rejection count was never retained in this proof.
        NativeMoEMovementArchive archive(geometry());
        archive.observe(41, 7, {}, {}, {}, receipt);
        MoEOptimizationStatus status{.authority = MoEOptimizationAuthority::Device,
            .completed_decision_windows = archive.completedDecisionWindows(), .last_decision = archive.lastDecision()};
        EXPECT_FALSE(status.last_decision->rejected_cycles_observed);
        EXPECT_TRUE(moeOptimizationStatusJson(status)["last_decision"]["rejected_cycles"].is_null());
        EXPECT_EQ(moeOptimizationStatusJson(status)["last_decision"]["rejected_cycles_observed"], false);
    }

}
