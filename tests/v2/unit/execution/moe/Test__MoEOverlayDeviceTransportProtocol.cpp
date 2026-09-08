/**
 * @file Test__MoEOverlayDeviceTransportProtocol.cpp
 * @brief Device-free proof of the device-authority/host-transport boundary.
 *
 * Synthetic mapped records exercise authentication and publication ordering
 * without starting a device. Diagnostic action-entry observations must never
 * stand in for the dedicated preparation, commit, or completion receipts.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEOverlayDeviceControllerGraphService.h"
#include "execution/moe/MoEOverlayDeviceControllerProtocol.h"
#include "execution/moe/MoEOverlayDeviceTransportProtocol.h"

#include <array>
#include <limits>
#include <memory>

namespace llaminar2
{
    namespace
    {
        constexpr std::uint64_t kFingerprint = 0x84b7a23du;

        /** @return Digest matching both device and host command validators. */
        std::uint64_t digest(
            const MoEOverlayDeviceMovementCommand &command) noexcept
        {
            std::uint64_t result = moeOverlayCommandDigestSeed(1u);
            const auto *words = reinterpret_cast<const std::uint64_t *>(
                &command);
            constexpr std::size_t kWords =
                sizeof(command) / sizeof(std::uint64_t);
            for (std::size_t index = 0u; index < kWords; ++index)
                result ^= moeOverlayCommandDigestWord(words[index], index);
            return result;
        }

        /** Complete one-group mapped record family with disjoint writers. */
        struct Fixture
        {
            Fixture()
                : device(
                      controller,
                      groups,
                      command)
            {
                MoEOverlayDeviceControllerProtocol::initialize(
                    controller,
                    groups,
                    command,
                    kFingerprint,
                    /*leader_group_id=*/0u,
                    /*leader_participant_id=*/0u,
                    /*initial_durable_epoch=*/7u,
                    roots);
                layout.topology_fingerprint = kFingerprint;
                layout.participant_count = 2u;
                layout.group_count = 1u;
                layout.num_layers = 2u;
                layout.num_experts = 8u;
                layout.command_capacity = entries.size();
                layout.mapping_bytes = 4096u;
                layout.payload_bytes_per_layer_offset = 512u;
                layout.minimum_window_activations = 1u;
                layout.maximum_cycles_per_wave = 1u;
                layout.payload_geometry_fingerprint = 1u;

                transport.group_id = 0u;
                transport.topology_fingerprint = kFingerprint;
                for (std::uint32_t participant = 0u;
                     participant < participant_records.size();
                     ++participant)
                {
                    participant_records[participant].participant_id =
                        participant;
                    participant_records[participant].group_id = 0u;
                    participant_records[participant].topology_fingerprint =
                        kFingerprint;
                }
                binding = {
                    .group_id = 0,
                    .root_world_rank = 0,
                    .topology_fingerprint = kFingerprint,
                    .command_capacity =
                        static_cast<std::uint32_t>(entries.size()),
                    .layout = &layout,
                    .controller = &controller,
                    .command = &command,
                    .command_entries = entries.data(),
                    .group_records = {groups.data()},
                    .topology_participant_records = {
                        participant_records.data()},
                    .topology_participant_record_counts = {
                        static_cast<std::uint32_t>(
                            participant_records.size())},
                    .group_record_count =
                        static_cast<std::uint32_t>(groups.size()),
                    .participant_records = participant_records.data(),
                    .participant_record_count = static_cast<std::uint32_t>(
                        participant_records.size()),
                    .transport = &transport,
                    .lifetime = std::static_pointer_cast<const void>(lifetime),
                };
            }

            /** Publish one valid physical Dynamic command from the device oracle. */
            std::uint64_t publishDynamic()
            {
                const auto transaction = device.beginTransaction(
                    MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement,
                    MoEOverlayDeviceDemandPhase::Decode);
                EXPECT_TRUE(transaction.has_value());
                entries[0] = {
                    .op = static_cast<std::uint32_t>(
                        MoEOverlayDeviceMovementOp::DurableMove),
                    .ordinal = 0u,
                    .layer = 1u,
                    .expert = 3u,
                    .source_participant = 0u,
                    .destination_participant = 1u,
                    .payload_slot = 0u,
                    .flags = static_cast<std::uint32_t>(
                        MoEOverlayDeviceMovementAxis::ParticipantPlacement),
                    .payload_bytes = 4096u,
                    .source_epoch = 7u,
                    .candidate_epoch = 8u,
                };
                EXPECT_TRUE(device.publishGroupSnapshot(
                    0u, *transaction, 0x1234u, 64u));
                EXPECT_TRUE(device.publishCommand(
                    *transaction,
                    1u,
                    digest(entries[0]),
                    entries[0].payload_bytes,
                    1u,
                    1u,
                    0u,
                    {
                        .snapshot_observations = 64u,
                        .priority_cost_before = 0u,
                        .priority_cost_after = 0u,
                        .same_priority_makespan_before = 100u,
                        .same_priority_makespan_after = 80u,
                        .accepted_cycles = 1u,
                        .same_priority_moves = 1u,
                        .changed_layers = 1u,
                        .projected_service_gain_ns = 1'000u,
                        .projected_transfer_and_repack_ns = 100u,
                        .projected_inference_interference_ns = 100u,
                        .projected_net_benefit_ns = 800u,
                    }));
                return *transaction;
            }

            MoEOverlayDeviceControllerFabricLayoutHeader layout;
            MoEOverlayDeviceControllerSharedHeader controller;
            std::array<MoEOverlayDeviceControllerGroupRecord, 1> groups;
            MoEOverlayDeviceControllerCommandHeader command;
            std::array<MoEOverlayDeviceMovementCommand, 4> entries;
            std::array<MoEOverlayDeviceControllerParticipantRecord, 2>
                participant_records;
            MoEOverlayDeviceControllerTransportRecord transport;
            std::array<std::uint32_t, 1> roots{0u};
            std::shared_ptr<int> lifetime = std::make_shared<int>(1);
            MoEOverlayDeviceControllerTransportBinding binding;
            MoEOverlayDeviceControllerProtocol device;
        };
    } // namespace

    /**
     * A follower timeout must identify a remote participant's missing receipt.
     * Action-entry words are observation only: describing them cannot advance
     * either group's dedicated preparation receipt.
     */
    TEST(MoEOverlayDeviceTransportProtocol,
         LifecycleDiagnosticIncludesRemoteParticipantsWithoutAdvancingThem)
    {
        Fixture fixture;
        auto follower_group = fixture.groups[0];
        follower_group.group_id = 1u;
        auto follower_participants = fixture.participant_records;
        for (auto &participant : follower_participants)
        {
            participant.group_id = 1u;
            participant.participant_id += 2u;
            participant.observed_action = static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerAction::AcknowledgePrepared);
            participant.observed_action_transaction = 28u;
            participant.prepared_transaction = 28u;
        }
        fixture.participant_records[0].observed_action =
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerAction::ApplyRuntimeCandidate);
        fixture.participant_records[0].observed_action_transaction = 28u;
        fixture.participant_records[0].prepared_transaction = 27u;
        fixture.layout.group_count = 2u;
        fixture.layout.participant_count = 4u;
        fixture.transport.group_id = 1u;
        fixture.binding.group_id = 1;
        fixture.binding.root_world_rank = 1;
        fixture.binding.group_record_count = 2u;
        fixture.binding.group_records[1] = &follower_group;
        fixture.binding.topology_participant_records[1] =
            follower_participants.data();
        fixture.binding.topology_participant_record_counts[1] =
            follower_participants.size();
        fixture.binding.participant_records = follower_participants.data();
        ASSERT_TRUE(fixture.binding.valid());
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);

        const auto description = transport.describeLifecycle();
        EXPECT_NE(description.find("participants=[group=0{0:status="),
                  std::string::npos) << description;
        EXPECT_NE(description.find("group=1{2:status="), std::string::npos)
            << description;
        EXPECT_NE(description.find("entry_transaction=28,snapshot=0,prepared=27"),
                  std::string::npos) << description;
        EXPECT_NE(description.find("entry_transaction=28,snapshot=0,prepared=28"),
                  std::string::npos) << description;
        EXPECT_EQ(fixture.participant_records[0].prepared_transaction, 27u);
        EXPECT_EQ(follower_participants[0].prepared_transaction, 28u);
        EXPECT_EQ(fixture.groups[0].prepared_transaction, 0u);
        EXPECT_EQ(follower_group.prepared_transaction, 0u);

        // Missing topology lanes are rejected before diagnostic traversal.
        fixture.binding.topology_participant_records[0] = nullptr;
        EXPECT_THROW(
            MoEOverlayDeviceTransportProtocol{fixture.binding},
            std::invalid_argument);
        fixture.groups[0].topology_fingerprint = 0u;
        EXPECT_EQ(transport.describeLifecycle(), "group=1,binding=invalid");
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         PhysicalEdgesCannotAdvanceBeforeDeviceAuthoredPhases)
    {
        Fixture fixture;
        ASSERT_TRUE(fixture.binding.valid());
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);

        EXPECT_EQ(
            transport.tryAcquire(0u).status,
            MoEOverlayDeviceTransportAcquireStatus::Waiting);
        const std::uint64_t transaction = fixture.publishDynamic();
        auto acquired = transport.tryAcquire(0u);
        ASSERT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Ready)
            << acquired.error;
        ASSERT_TRUE(acquired.batch.valid());
        EXPECT_TRUE(acquired.batch.movesWeights());
        EXPECT_EQ(acquired.batch.header.transaction_id, transaction);
        EXPECT_FALSE(transport.commitRequested(acquired.batch));
        EXPECT_FALSE(transport.preparationReady(acquired.batch));

        // A started (or apparently completed) action is diagnostic evidence,
        // not permission to consume an unpublished runtime or reuse its inbox.
        for (auto &participant : fixture.participant_records)
        {
            participant.observed_action = static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerAction::CompleteDynamicRetirement);
            participant.observed_action_transaction = transaction;
        }
        EXPECT_FALSE(transport.preparationReady(acquired.batch));
        EXPECT_FALSE(transport.commitRequested(acquired.batch));
        EXPECT_FALSE(transport.transactionComplete(acquired.batch));
        EXPECT_NE(transport.describeLifecycle().find("entered_action="),
                  std::string::npos);

        std::string error;
        ASSERT_TRUE(transport.publishPrepared(acquired.batch, &error)) << error;
        EXPECT_FALSE(transport.preparationReady(acquired.batch));
        for (auto &participant : fixture.participant_records)
            participant.prepared_transaction = transaction;
        EXPECT_TRUE(transport.preparationReady(acquired.batch));
        /* The physical candidate receipt precedes the bounded RCU graph. It
         * cannot select E+1: commit and every participant publication remain
         * device-authored below. */
        ASSERT_TRUE(transport.publishPublished(acquired.batch, &error)) << error;
        EXPECT_EQ(fixture.groups[0].prepared_transaction, 0u);
        ASSERT_TRUE(fixture.device.acknowledgePrepared(0u, transaction));
        EXPECT_TRUE(transport.allGroupsPrepared(acquired.batch));
        ASSERT_TRUE(fixture.device.beginCommit(transaction));
        ASSERT_TRUE(transport.commitRequested(acquired.batch));
        EXPECT_FALSE(transport.publicationReady(acquired.batch));
        for (auto &participant : fixture.participant_records)
            participant.published_transaction = transaction;
        EXPECT_TRUE(transport.publicationReady(acquired.batch));
        EXPECT_TRUE(transport.groupPublicationReady(acquired.batch));
        EXPECT_EQ(fixture.groups[0].published_transaction, 0u);
        ASSERT_TRUE(fixture.device.acknowledgePublished(0u, transaction));
        EXPECT_TRUE(transport.allGroupsPublished(acquired.batch));
        ASSERT_TRUE(fixture.device.publishAdmission(transaction));
        ASSERT_TRUE(fixture.device.beginDynamicRetirement(transaction));
        EXPECT_TRUE(transport.retirementOpen(acquired.batch));
        EXPECT_FALSE(transport.runtimeReadersReady(acquired.batch));
        fixture.participant_records[0].retirement_ready_epoch = 7u;
        EXPECT_FALSE(transport.runtimeReadersReady(acquired.batch));
        fixture.participant_records[1].retirement_ready_epoch = 7u;
        EXPECT_TRUE(transport.runtimeReadersReady(acquired.batch));
        EXPECT_FALSE(transport.retirementRequested(acquired.batch));
        fixture.participant_records[0].retired_epoch = 7u;
        EXPECT_FALSE(transport.retirementRequested(acquired.batch));
        fixture.participant_records[1].retired_epoch = 7u;
        ASSERT_TRUE(transport.retirementRequested(acquired.batch));
        EXPECT_FALSE(transport.groupRetirementReady(acquired.batch));
        ASSERT_TRUE(transport.publishRetired(acquired.batch, &error)) << error;
        EXPECT_TRUE(transport.groupRetirementReady(acquired.batch));
        EXPECT_EQ(fixture.groups[0].retired_epoch, 0u);
        ASSERT_TRUE(fixture.device.acknowledgeRetired(
            0u, transaction, /*retired_epoch=*/7u));
        EXPECT_EQ(fixture.groups[0].retired_epoch, 7u);
        EXPECT_TRUE(transport.allGroupsRetired(acquired.batch));
        ASSERT_TRUE(fixture.device.completeDynamicRetirement(transaction));
        EXPECT_TRUE(transport.transactionComplete(acquired.batch));
        EXPECT_EQ(fixture.controller.completed_transaction, transaction);

        EXPECT_EQ(fixture.device.currentDurableEpoch(), 8u);
        EXPECT_EQ(
            fixture.device.state(),
            MoEOverlayDeviceControllerState::Complete);
        EXPECT_EQ(
            fixture.transport.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransportState::Retired));
        EXPECT_EQ(fixture.transport.command_transaction, transaction);
        EXPECT_EQ(fixture.transport.prepared_transaction, transaction);
        EXPECT_EQ(fixture.transport.published_transaction, transaction);
        EXPECT_EQ(fixture.transport.retired_epoch, 7u);
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         LateFollowerAcceptsDurableReceiptsAfterAuthorityAdvances)
    {
        Fixture fixture;
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const std::uint64_t first = fixture.publishDynamic();
        auto acquired = transport.tryAcquire(0u);
        ASSERT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Ready)
            << acquired.error;

        std::string error;
        ASSERT_TRUE(transport.publishPrepared(acquired.batch, &error)) << error;
        for (auto &participant : fixture.participant_records)
        {
            participant.prepared_transaction = first;
            participant.published_transaction = first;
            participant.retirement_ready_epoch = 7u;
            participant.retired_epoch = 7u;
        }
        ASSERT_TRUE(transport.publishPublished(acquired.batch, &error)) << error;
        ASSERT_TRUE(fixture.device.acknowledgePrepared(0u, first));
        ASSERT_TRUE(fixture.device.beginCommit(first));
        ASSERT_TRUE(fixture.device.acknowledgePublished(0u, first));
        ASSERT_TRUE(fixture.device.publishAdmission(first));
        ASSERT_TRUE(fixture.device.beginDynamicRetirement(first));
        ASSERT_TRUE(transport.publishRetired(acquired.batch, &error)) << error;
        ASSERT_TRUE(fixture.device.acknowledgeRetired(0u, first, 7u));
        ASSERT_TRUE(fixture.device.completeDynamicRetirement(first));

        // Simulate the authority rank beginning N+1 before this rank-local
        // scheduler observes the final N fan-in and completion predicates.
        const auto second = fixture.device.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            MoEOverlayDeviceDemandPhase::Decode);
        ASSERT_TRUE(second.has_value());
        ASSERT_GT(*second, first);
        EXPECT_EQ(
            fixture.device.state(),
            MoEOverlayDeviceControllerState::CollectingSnapshots);

        EXPECT_TRUE(transport.allGroupsPrepared(acquired.batch));
        EXPECT_TRUE(transport.allGroupsPublished(acquired.batch));
        EXPECT_TRUE(transport.allGroupsRetired(acquired.batch));
        EXPECT_TRUE(transport.transactionComplete(acquired.batch));
    }

    /**
     * Completion does not imply that every transport worker acquired an empty
     * command. Opening the next snapshot must preserve that immutable command
     * while publishing the next transaction's independent phase intent.
     */
    TEST(MoEOverlayDeviceTransportProtocol,
         NextSnapshotRetainsUnacquiredEmptyCommandAndIndependentPhase)
    {
        Fixture fixture;
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const auto first = fixture.device.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            MoEOverlayDeviceDemandPhase::Prefill);
        ASSERT_TRUE(first.has_value());
        ASSERT_TRUE(fixture.device.publishGroupSnapshot(0u, *first, 0x9911u, 64u));
        ASSERT_TRUE(fixture.device.publishCommand(
            *first, 0u, moeOverlayCommandDigestSeed(0u), 0u, 0u, 0u, 0u));
        const auto sealed = fixture.command;

        // Complete the CPU specification without acquiring the host transport
        // view. The captured device empty-policy path reaches the same terminal
        // receipt directly, so a slow host is allowed to arrive after it.
        ASSERT_TRUE(fixture.device.acknowledgePrepared(0u, *first));
        ASSERT_TRUE(fixture.device.beginCommit(*first));
        ASSERT_TRUE(fixture.device.acknowledgePublished(0u, *first));
        ASSERT_TRUE(fixture.device.publishAdmission(*first));
        ASSERT_TRUE(fixture.device.beginDynamicRetirement(*first));
        ASSERT_TRUE(fixture.device.acknowledgeRetired(0u, *first, 7u));
        ASSERT_TRUE(fixture.device.completeDynamicRetirement(*first));
        const auto next = fixture.device.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            MoEOverlayDeviceDemandPhase::Decode);
        ASSERT_TRUE(next.has_value());
        ASSERT_FALSE(fixture.device.allSnapshotsReady(*next));

        const auto acquired = transport.tryAcquire(0u);
        ASSERT_EQ(acquired.status, MoEOverlayDeviceTransportAcquireStatus::Ready)
            << acquired.error;
        EXPECT_EQ(acquired.batch.header.transaction_id, *first);
        EXPECT_EQ(acquired.batch.header.demand_phase, sealed.demand_phase);
        EXPECT_EQ(acquired.batch.header.command_digest, sealed.command_digest);
        EXPECT_TRUE(transport.transactionComplete(acquired.batch));

        std::uint64_t observed = 0u;
        auto kind = MoEOverlayDeviceControllerTransactionKind::Invalid;
        auto phase = MoEOverlayDeviceDemandPhase::Invalid;
        ASSERT_TRUE(transport.snapshotTransactionAfter(*first, &observed, &kind, &phase));
        EXPECT_EQ(observed, *next);
        EXPECT_EQ(phase, MoEOverlayDeviceDemandPhase::Decode);
        EXPECT_EQ(transport.tryAcquire(*first).status,
                  MoEOverlayDeviceTransportAcquireStatus::Waiting);
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         BoundedSnapshotSchedulerObservesOnlyMonotonicLifecycleTickets)
    {
        Fixture fixture;
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const auto transaction = fixture.device.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            MoEOverlayDeviceDemandPhase::Prefill);
        ASSERT_TRUE(transaction.has_value());

        std::uint64_t observed = 0u;
        MoEOverlayDeviceControllerTransactionKind observed_kind =
            MoEOverlayDeviceControllerTransactionKind::Invalid;
        MoEOverlayDeviceDemandPhase observed_phase =
            MoEOverlayDeviceDemandPhase::Invalid;
        EXPECT_TRUE(transport.snapshotTransactionAfter(
            0u, &observed, &observed_kind, &observed_phase));
        EXPECT_EQ(observed, *transaction);
        EXPECT_EQ(
            observed_kind,
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement);
        EXPECT_EQ(observed_phase, MoEOverlayDeviceDemandPhase::Prefill);
        EXPECT_FALSE(transport.snapshotTransactionAfter(
            *transaction, &observed, &observed_kind, &observed_phase));
        EXPECT_FALSE(transport.localSnapshotsReady(*transaction));

        fixture.participant_records[0].snapshot_transaction = *transaction;
        EXPECT_FALSE(transport.localSnapshotsReady(*transaction));
        fixture.participant_records[1].snapshot_transaction = *transaction;
        EXPECT_TRUE(transport.localSnapshotsReady(*transaction));
        EXPECT_FALSE(transport.allGroupsSnapshotted(*transaction));

        ASSERT_TRUE(fixture.device.publishGroupSnapshot(
            0u, *transaction, 0x1234u, 64u));
        EXPECT_TRUE(transport.allGroupsSnapshotted(*transaction));

        // Command publication closes collection. The live scheduling predicates
        // reject another participant-snapshot phase, while the group fan-in
        // remains a durable receipt for a slower rank-local scheduler.
        fixture.entries[0] = {
            .op = static_cast<std::uint32_t>(
                MoEOverlayDeviceMovementOp::DurableMove),
            .ordinal = 0u,
            .layer = 1u,
            .expert = 3u,
            .source_participant = 0u,
            .destination_participant = 1u,
            .payload_slot = 0u,
            .payload_bytes = 4096u,
            .source_epoch = 7u,
            .candidate_epoch = 8u,
        };
        ASSERT_TRUE(fixture.device.publishCommand(
            *transaction,
            1u,
            digest(fixture.entries[0]),
            fixture.entries[0].payload_bytes,
            1u,
            1u,
            0u));
        EXPECT_FALSE(transport.snapshotTransactionAfter(
            0u, &observed, &observed_kind, &observed_phase));
        EXPECT_FALSE(transport.localSnapshotsReady(*transaction));
        EXPECT_TRUE(transport.allGroupsSnapshotted(*transaction));
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         PreparedContextRestoreTicketAndZeroCommandFormAnExactTerminalProof)
    {
        Fixture fixture;
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const auto transaction = fixture.device.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::
                PreparedContextRestore,
            MoEOverlayDeviceDemandPhase::Invalid);
        ASSERT_TRUE(transaction.has_value());

        std::uint64_t observed_transaction = 0u;
        MoEOverlayDeviceControllerTransactionKind observed_kind =
            MoEOverlayDeviceControllerTransactionKind::Invalid;
        MoEOverlayDeviceDemandPhase observed_phase =
            MoEOverlayDeviceDemandPhase::Decode;
        ASSERT_TRUE(transport.snapshotTransactionAfter(
            0u,
            &observed_transaction,
            &observed_kind,
            &observed_phase));
        EXPECT_EQ(observed_transaction, *transaction);
        EXPECT_EQ(
            observed_kind,
            MoEOverlayDeviceControllerTransactionKind::
                PreparedContextRestore);
        EXPECT_EQ(observed_phase, MoEOverlayDeviceDemandPhase::Invalid);

        ASSERT_TRUE(fixture.device.publishGroupSnapshot(
            0u, *transaction, 0x9911u, 0u));
        ASSERT_TRUE(fixture.device.publishCommand(
            *transaction,
            0u,
            moeOverlayCommandDigestSeed(0u),
            0u,
            0u,
            0u,
            0u));
        auto acquired = transport.tryAcquire(0u);
        ASSERT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Ready)
            << acquired.error;
        EXPECT_TRUE(acquired.batch.valid());
        EXPECT_FALSE(acquired.batch.movesWeights());
        EXPECT_EQ(
            acquired.batch.header.candidate_epoch,
            acquired.batch.header.base_epoch);
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         TamperedCommandFailsWithoutPublishingPhysicalReadiness)
    {
        Fixture fixture;
        const std::uint64_t transaction = fixture.publishDynamic();
        fixture.entries[0].expert = 4u; // Digest no longer authenticates bytes.

        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const auto acquired = transport.tryAcquire(0u);
        EXPECT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Failed);
        EXPECT_FALSE(acquired.error.empty());
        EXPECT_EQ(fixture.transport.command_transaction, 0u);
        EXPECT_EQ(fixture.transport.prepared_transaction, 0u);
        EXPECT_EQ(
            fixture.transport.status_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::PhysicalTransportFailure));
        EXPECT_EQ(fixture.controller.transaction_id, transaction);
        EXPECT_EQ(fixture.controller.current_durable_epoch, 7u);
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         AuthorityFailureBeforeCommandPublicationFailsImmediately)
    {
        Fixture fixture;
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);

        // An invalid commit makes the device authority terminal before it can
        // publish immutable command bytes. The host follower must report that
        // exact terminal edge on its first poll, not wait for a command that
        // can no longer exist and eventually report a timeout.
        ASSERT_FALSE(fixture.device.beginCommit(/*transaction=*/1u));
        ASSERT_EQ(
            fixture.device.state(),
            MoEOverlayDeviceControllerState::Error);
        ASSERT_EQ(fixture.controller.command_transaction, 0u);

        const auto acquired = transport.tryAcquire(0u);
        EXPECT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Failed);
        EXPECT_NE(acquired.error.find("authority is terminal"),
                  std::string::npos);
        EXPECT_NE(acquired.error.find("error_code="), std::string::npos);

        // Observing an authority failure does not relabel it as a physical
        // transport failure or mutate the group-local completion lane.
        EXPECT_EQ(
            fixture.transport.status_code,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::None));
        EXPECT_EQ(
            fixture.transport.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerTransportState::Idle));
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         BackgroundWorkerObservesOnlyExactAuthorityRejection)
    {
        Fixture fixture;
        const std::uint64_t transaction = fixture.publishDynamic();
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const auto acquired = transport.tryAcquire(0u);
        ASSERT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Ready)
            << acquired.error;
        EXPECT_FALSE(transport.authorityRejected(acquired.batch));

        // Opening commit before group preparation is a fatal authority-side
        // lifecycle violation. The transport may observe that terminal edge,
        // but receives no placement, histogram, or repair capability.
        ASSERT_FALSE(fixture.device.beginCommit(transaction));
        EXPECT_EQ(
            fixture.device.state(),
            MoEOverlayDeviceControllerState::Error);
        EXPECT_TRUE(transport.authorityRejected(acquired.batch));

        auto foreign = acquired.batch;
        foreign.header.transaction_id += 1u;
        EXPECT_FALSE(transport.authorityRejected(foreign));
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         ParticipantFailureIsVisibleBeforeDeferredGroupFanIn)
    {
        Fixture fixture;
        const std::uint64_t transaction = fixture.publishDynamic();
        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const auto acquired = transport.tryAcquire(0u);
        ASSERT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Ready)
            << acquired.error;
        ASSERT_EQ(fixture.controller.transaction_id, transaction);
        ASSERT_NE(
            fixture.controller.state,
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerState::Error));
        EXPECT_FALSE(transport.authorityRejected(acquired.batch));

        // A participant can fail before its group root reaches the next fan-in
        // action. The physical worker must unwind immediately instead of
        // waiting for the global leader to rediscover the same error later.
        fixture.participant_records[1].status_code =
            static_cast<std::uint32_t>(
                MoEOverlayDeviceControllerError::InvalidState);
        EXPECT_TRUE(transport.authorityRejected(acquired.batch));

        auto foreign = acquired.batch;
        foreign.header.transaction_id += 1u;
        EXPECT_FALSE(transport.authorityRejected(foreign));
    }

    TEST(MoEOverlayDeviceTransportProtocol,
         RemoteGroupFailureIsVisibleBeforeTopologyFanInDeadline)
    {
        Fixture fixture;
        const std::uint64_t transaction = fixture.publishDynamic();
        MoEOverlayDeviceControllerGroupRecord remote_group;
        remote_group.group_id = 1u;
        remote_group.root_participant_id = 2u;
        remote_group.topology_fingerprint = kFingerprint;
        std::array<MoEOverlayDeviceControllerParticipantRecord, 1>
            remote_participants;
        remote_participants[0].participant_id = 2u;
        remote_participants[0].group_id = 1u;
        remote_participants[0].topology_fingerprint = kFingerprint;
        fixture.layout.group_count = 2u;
        fixture.layout.participant_count = 3u;
        fixture.binding.group_records[1] = &remote_group;
        fixture.binding.topology_participant_records[1] =
            remote_participants.data();
        fixture.binding.topology_participant_record_counts[1] =
            static_cast<std::uint32_t>(remote_participants.size());
        fixture.binding.group_record_count = 2u;

        MoEOverlayDeviceTransportProtocol transport(fixture.binding);
        const auto acquired = transport.tryAcquire(0u);
        ASSERT_EQ(
            acquired.status,
            MoEOverlayDeviceTransportAcquireStatus::Ready)
            << acquired.error;
        ASSERT_EQ(fixture.controller.transaction_id, transaction);
        EXPECT_FALSE(transport.authorityRejected(acquired.batch));

        // The local transport lane is healthy. A remote group-root failure
        // must nevertheless stop a topology-wide fan-in without waiting for
        // the authority leader to rediscover and copy that terminal code.
        remote_group.status_code = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerError::InvalidState);
        EXPECT_TRUE(transport.authorityRejected(acquired.batch));
    }

    TEST(MoEOverlayMaintenanceBoundaryGate,
         RequiresPhasePureCompleteWindowsAndCoalescesOnlyAfterward)
    {
        EXPECT_THROW(
            MoEOverlayMaintenanceBoundaryGate(0u), std::invalid_argument);
        MoEOverlayMaintenanceBoundaryGate gate(64u);
        gate.notify(MoEOverlayInferencePhase::Prefill, 63u);
        EXPECT_FALSE(gate.ready());
        EXPECT_FALSE(gate.consumeReady());

        // Decode cannot complete a partial prefill window. It is admitted as
        // its own immutable scheduler ticket while prefill remains pending.
        gate.notify(MoEOverlayInferencePhase::Decode, 64u);
        EXPECT_TRUE(gate.ready());

        // Device placement remains closed until immutable measured economics
        // are published. Polling while closed must preserve every completed
        // token so certification can overlap inference without losing the
        // first eligible maintenance epoch.
        EXPECT_FALSE(gate.consumeReady(/*admission_open=*/false));
        EXPECT_TRUE(gate.ready());
        const auto decode = gate.consumeReady();
        ASSERT_TRUE(decode);
        EXPECT_EQ(decode.phase, MoEOverlayInferencePhase::Decode);
        EXPECT_EQ(decode.completed_tokens, 64u);
        EXPECT_FALSE(gate.ready());

        gate.notify(MoEOverlayInferencePhase::Prefill, 1u);
        const auto prefill = gate.consumeReady();
        ASSERT_TRUE(prefill);
        EXPECT_EQ(prefill.phase, MoEOverlayInferencePhase::Prefill);
        EXPECT_EQ(prefill.completed_tokens, 64u);

        // A worker busy with one movement wave may receive more than a single
        // window. All of that histogram evidence belongs to one fresh
        // observation; it must not trigger a burst of back-to-back epochs.
        gate.notify(MoEOverlayInferencePhase::Decode, 70u);
        const auto coalesced = gate.consumeReady();
        ASSERT_TRUE(coalesced);
        EXPECT_EQ(coalesced.phase, MoEOverlayInferencePhase::Decode);
        EXPECT_EQ(coalesced.completed_tokens, 70u);
        EXPECT_FALSE(gate.consumeReady());

        // The production controller grows cadence after a completed attempt.
        // Tokens retired while that attempt ran are not discarded, but they
        // must satisfy the new cooldown before another wave is admitted.
        gate.notify(MoEOverlayInferencePhase::Decode, 70u);
        EXPECT_EQ(
            gate.advanceAfterReceipt(
                128u,
                1.5,
                MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(
                        /*snapshot_observations=*/0u,
                        /*command_count=*/0u)),
            64u);
        EXPECT_EQ(gate.requiredTokens(), 64u);
        EXPECT_TRUE(gate.ready());

        // The calibration rebase can leave a cadence-only notification that
        // produces an empty first transaction. It must not consume the one
        // adaptive-window transition reserved for real demand evidence.
        const auto empty_post_rebase = gate.consumeReady();
        ASSERT_TRUE(empty_post_rebase);
        EXPECT_EQ(empty_post_rebase.completed_tokens, 70u);
        gate.notify(MoEOverlayInferencePhase::Decode, 70u);
        EXPECT_EQ(
            gate.advanceAfterReceipt(
                128u,
                1.5,
                MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(
                        /*snapshot_observations=*/1u,
                        /*command_count=*/0u)),
            96u);
        EXPECT_EQ(gate.requiredTokens(), 96u);
        EXPECT_FALSE(gate.ready());
        gate.notify(MoEOverlayInferencePhase::Decode, 26u);
        const auto grown_window = gate.consumeReady();
        ASSERT_TRUE(grown_window);
        EXPECT_EQ(grown_window.phase, MoEOverlayInferencePhase::Decode);
        EXPECT_EQ(grown_window.completed_tokens, 96u);
        EXPECT_EQ(
            gate.advanceAfterReceipt(
                128u,
                1.5,
                MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(1u, 0u)),
            128u);
        EXPECT_EQ(
            gate.advanceAfterReceipt(
                128u,
                2.0,
                MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(1u, 0u)),
            128u);

        // A malformed zero-row progress report cannot create an epoch, and a
        // huge bounded prefill report saturates rather than wrapping the gate.
        gate.notify(MoEOverlayInferencePhase::Prefill, 0u);
        EXPECT_FALSE(gate.ready());
        gate.notify(
            MoEOverlayInferencePhase::Prefill,
            std::numeric_limits<std::uint64_t>::max());
        gate.notify(MoEOverlayInferencePhase::Prefill, 1u);
        const auto saturated = gate.consumeReady();
        ASSERT_TRUE(saturated);
        EXPECT_EQ(saturated.phase, MoEOverlayInferencePhase::Prefill);
        EXPECT_EQ(
            saturated.completed_tokens,
            std::numeric_limits<std::uint64_t>::max());
    }

    TEST(MoEOverlayMaintenanceBoundaryGate,
         RetainsRapidCadenceUntilDevicePublishesObservedNoMovement)
    {
        MoEOverlayMaintenanceBoundaryGate gate(9u);

        // A tier-residency cycle changes the placement seen by the next policy
        // scan. Cooling down here used to strand participant-skew work behind
        // a 4096-token window in finite production parity requests.
        gate.notify(MoEOverlayInferencePhase::Prefill, 9u);
        ASSERT_TRUE(gate.consumeReady());
        EXPECT_EQ(
            gate.advanceAfterReceipt(
                4096u,
                4096.0 / 9.0,
                MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(
                        /*snapshot_observations=*/72u,
                        /*command_count=*/3u)),
            9u);

        // A second independently useful movement remains part of the same
        // convergence burst, regardless of which axis the first wave covered.
        gate.notify(MoEOverlayInferencePhase::Prefill, 9u);
        ASSERT_TRUE(gate.consumeReady());
        EXPECT_EQ(
            gate.advanceAfterReceipt(
                4096u,
                4096.0 / 9.0,
                MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(
                        /*snapshot_observations=*/72u,
                        /*command_count=*/2u)),
            9u);

        // Only an authenticated policy pass that consumed demand and emitted
        // no command certifies that the burst can enter its long cooldown.
        gate.notify(MoEOverlayInferencePhase::Prefill, 9u);
        ASSERT_TRUE(gate.consumeReady());
        EXPECT_EQ(
            gate.advanceAfterReceipt(
                4096u,
                4096.0 / 9.0,
                MoEOverlayMaintenanceCadenceReceipt::
                    fromDynamicTransaction(
                        /*snapshot_observations=*/72u,
                        /*command_count=*/0u)),
            4096u);
    }

    TEST(MoEOverlayMaintenanceBoundaryGate,
         AuthoritySelectedPhaseLeavesOtherReadyWindowPending)
    {
        MoEOverlayMaintenanceBoundaryGate gate(8u);
        gate.notify(MoEOverlayInferencePhase::Prefill, 13u);
        gate.notify(MoEOverlayInferencePhase::Decode, 11u);

        ASSERT_TRUE(gate.ready());
        ASSERT_TRUE(
            gate.readyForPhase(MoEOverlayInferencePhase::Prefill));
        ASSERT_TRUE(
            gate.readyForPhase(MoEOverlayInferencePhase::Decode));

        const auto decode = gate.consumeReadyForPhase(
            MoEOverlayInferencePhase::Decode);
        ASSERT_TRUE(decode);
        EXPECT_EQ(decode.phase, MoEOverlayInferencePhase::Decode);
        EXPECT_EQ(decode.completed_tokens, 11u);

        // A follower joining a decode ticket must not consume an older
        // prefill sideband merely because both phases are ready.
        EXPECT_FALSE(
            gate.readyForPhase(MoEOverlayInferencePhase::Decode));
        EXPECT_TRUE(
            gate.readyForPhase(MoEOverlayInferencePhase::Prefill));
        EXPECT_TRUE(gate.ready());

        EXPECT_FALSE(gate.consumeReadyForPhase(
            MoEOverlayInferencePhase::Prefill,
            /*admission_open=*/false));
        const auto prefill = gate.consumeReadyForPhase(
            MoEOverlayInferencePhase::Prefill);
        ASSERT_TRUE(prefill);
        EXPECT_EQ(prefill.completed_tokens, 13u);
        EXPECT_FALSE(gate.ready());
    }
} // namespace llaminar2
