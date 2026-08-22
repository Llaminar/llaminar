/**
 * @file Test__MoEOverlayDeviceControllerProtocol.cpp
 * @brief Adversarial CPU tests for the shared GPU controller lifecycle ABI.
 *
 * These tests execute the same acquire/release state machine that CUDA and HIP
 * controller kernels consume. They deliberately use a variable number of
 * independently publishing groups and must remain device-free and fast.
 */

#include "execution/moe/MoEOverlayDeviceControllerProtocol.h"
#include "execution/moe/MoEOverlayWorkerDrainProtocol.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <thread>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** Own one pristine shared record family and its bound protocol. */
        struct ProtocolFixture
        {
            explicit ProtocolFixture(std::size_t group_count)
                : groups(group_count), roots(group_count),
                  protocol(header, groups, command)
            {
                for (std::size_t index = 0; index < roots.size(); ++index)
                    roots[index] = static_cast<std::uint32_t>(100u + index);
                MoEOverlayDeviceControllerProtocol::initialize(
                    header,
                    groups,
                    command,
                    0x0123456789abcdefULL,
                    0u,
                    roots.front(),
                    7u,
                    roots);
            }

            MoEOverlayDeviceControllerSharedHeader header;
            std::vector<MoEOverlayDeviceControllerGroupRecord> groups;
            MoEOverlayDeviceControllerCommandHeader command;
            std::vector<std::uint32_t> roots;
            MoEOverlayDeviceControllerProtocol protocol;
        };

        /** Publish every group edge in a deliberately non-numeric order. */
        template <typename Publish>
        void publishReverse(std::size_t count, Publish &&publish)
        {
            for (std::size_t offset = 0; offset < count; ++offset)
            {
                const std::uint32_t group =
                    static_cast<std::uint32_t>(count - offset - 1u);
                ASSERT_TRUE(publish(group));
            }
        }

        /** Drive snapshot fan-in and publish one immutable command. */
        std::uint64_t prepareCommand(
            ProtocolFixture &fixture,
            MoEOverlayDeviceControllerTransactionKind kind,
            std::uint32_t command_count,
            std::uint64_t packed_weight_bytes)
        {
            const auto transaction = fixture.protocol.beginTransaction(
                kind,
                kind == MoEOverlayDeviceControllerTransactionKind::
                            DynamicPlacement
                    ? MoEOverlayDeviceDemandPhase::Decode
                    : MoEOverlayDeviceDemandPhase::Invalid);
            EXPECT_TRUE(transaction.has_value());
            if (!transaction)
                return 0u;
            publishReverse(
                fixture.groups.size(),
                [&](std::uint32_t group)
                {
                    return fixture.protocol.publishGroupSnapshot(
                        group,
                        *transaction,
                        0x1000u + group,
                        200u + group);
                });
            EXPECT_TRUE(fixture.protocol.allSnapshotsReady(*transaction));
            EXPECT_TRUE(fixture.protocol.publishCommand(
                *transaction,
                command_count,
                0xfeed0000u + *transaction,
                packed_weight_bytes,
                command_count,
                command_count == 0u ? 0u : 1u,
                0u));
            return *transaction;
        }

        /** Drive prepare, commit, local publication, and global admission. */
        void admit(ProtocolFixture &fixture, std::uint64_t transaction)
        {
            publishReverse(
                fixture.groups.size(),
                [&](std::uint32_t group)
                {
                    return fixture.protocol.acknowledgePrepared(
                        group, transaction);
                });
            ASSERT_TRUE(fixture.protocol.allGroupsPrepared(transaction));
            ASSERT_TRUE(fixture.protocol.beginCommit(transaction));
            publishReverse(
                fixture.groups.size(),
                [&](std::uint32_t group)
                {
                    return fixture.protocol.acknowledgePublished(
                        group, transaction);
                });
            ASSERT_TRUE(fixture.protocol.allGroupsPublished(transaction));
            ASSERT_TRUE(fixture.protocol.publishAdmission(transaction));
        }
    } // namespace

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         DynamicPublishesGloballyBeforeRetiringEveryPriorEpoch)
    {
        ProtocolFixture fixture(3u);
        const auto transaction = prepareCommand(
            fixture,
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            2u,
            8u * 1024u * 1024u);

        EXPECT_EQ(fixture.protocol.currentDurableEpoch(), 7u);
        admit(fixture, transaction);
        EXPECT_EQ(fixture.protocol.state(),
                  MoEOverlayDeviceControllerState::Admitted);
        EXPECT_EQ(fixture.protocol.currentDurableEpoch(), 8u);
        EXPECT_EQ(fixture.header.admission_epoch, 8u);

        ASSERT_TRUE(fixture.protocol.beginDynamicRetirement(transaction));
        EXPECT_TRUE(fixture.protocol.acknowledgeRetired(2u, transaction, 7u));
        EXPECT_TRUE(fixture.protocol.acknowledgeRetired(0u, transaction, 7u));
        EXPECT_TRUE(fixture.protocol.acknowledgeRetired(1u, transaction, 7u));
        EXPECT_TRUE(
            fixture.protocol.completeDynamicRetirement(transaction));
        EXPECT_EQ(fixture.protocol.state(),
                  MoEOverlayDeviceControllerState::Complete);
        EXPECT_EQ(fixture.header.completed_transaction, transaction);
        EXPECT_EQ(fixture.protocol.error(),
                  MoEOverlayDeviceControllerError::None);
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         StaleIdleDrainAcknowledgementCannotSatisfyLaterShutdown)
    {
        MoEOverlayWorkerDrainProtocol drain;

        const auto first = drain.request();
        ASSERT_TRUE(first.valid());
        drain.acknowledge(first);
        ASSERT_TRUE(drain.acknowledged(first));

        /* Model the adversarial edge from production: the worker was idle and
         * acknowledged generation N, then selected work immediately before
         * the owner published shutdown generation N+1. */
        const auto second = drain.request();
        ASSERT_GT(second.generation, first.generation);
        EXPECT_TRUE(drain.shutdownRequested());
        EXPECT_TRUE(drain.acknowledgementPending());
        EXPECT_FALSE(drain.acknowledged(second));

        // Re-publishing the stale generation must not advance quiescence.
        drain.acknowledge(first);
        EXPECT_FALSE(drain.acknowledged(second));
        drain.acknowledge(second);
        EXPECT_TRUE(drain.acknowledged(second));
        EXPECT_FALSE(drain.acknowledgementPending());
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         StaticCompletesWithoutMovementOrEpochChange)
    {
        ProtocolFixture fixture(2u);
        const auto transaction = prepareCommand(
            fixture,
            MoEOverlayDeviceControllerTransactionKind::StaticCheck,
            0u,
            0u);
        admit(fixture, transaction);

        EXPECT_EQ(fixture.protocol.state(),
                  MoEOverlayDeviceControllerState::Complete);
        EXPECT_EQ(fixture.protocol.currentDurableEpoch(), 7u);
        EXPECT_EQ(fixture.header.admission_epoch, 7u);
        EXPECT_EQ(fixture.command.command_count, 0u);
        EXPECT_EQ(fixture.command.packed_weight_bytes, 0u);
        EXPECT_EQ(fixture.command.parallel_command_count, 0u);
        EXPECT_EQ(fixture.command.movement_round_count, 0u);
        EXPECT_EQ(fixture.command.hazard_count, 0u);
        EXPECT_EQ(fixture.protocol.activeLLEPTransaction(), 0u);
        EXPECT_EQ(fixture.header.completed_transaction, transaction);
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         StaticRejectsAnyClaimedMovement)
    {
        ProtocolFixture fixture(2u);
        const auto transaction = fixture.protocol.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::StaticCheck,
            MoEOverlayDeviceDemandPhase::Invalid);
        ASSERT_TRUE(transaction.has_value());
        ASSERT_TRUE(fixture.protocol.publishGroupSnapshot(
            0u, *transaction, 0x11u, 1u));
        ASSERT_TRUE(fixture.protocol.publishGroupSnapshot(
            1u, *transaction, 0x22u, 1u));

        EXPECT_FALSE(fixture.protocol.publishCommand(
            *transaction, 1u, 0x33u, 4096u, 1u, 1u, 0u));
        EXPECT_EQ(fixture.protocol.state(),
                  MoEOverlayDeviceControllerState::Error);
        EXPECT_EQ(fixture.protocol.error(),
                  MoEOverlayDeviceControllerError::InvalidCommand);
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         DynamicRejectsSerializedOrConflictingMovementWaves)
    {
        const auto reject = [](
                                std::uint32_t parallel_commands,
                                std::uint32_t movement_rounds,
                                std::uint32_t hazards)
        {
            ProtocolFixture fixture(3u);
            const auto transaction = fixture.protocol.beginTransaction(
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
                MoEOverlayDeviceDemandPhase::Decode);
            EXPECT_TRUE(transaction.has_value());
            if (!transaction)
                return;
            for (std::uint32_t group = 0u; group < 3u; ++group)
            {
                EXPECT_TRUE(fixture.protocol.publishGroupSnapshot(
                    group, *transaction, 0x500u + group, 32u));
            }
            EXPECT_FALSE(fixture.protocol.publishCommand(
                *transaction,
                4u,
                0xbeefu,
                16384u,
                parallel_commands,
                movement_rounds,
                hazards));
            EXPECT_EQ(
                fixture.protocol.error(),
                MoEOverlayDeviceControllerError::InvalidCommand);
        };

        reject(/*parallel_commands=*/3u, /*movement_rounds=*/1u, 0u);
        reject(/*parallel_commands=*/4u, /*movement_rounds=*/2u, 0u);
        reject(/*parallel_commands=*/4u, /*movement_rounds=*/1u, 1u);
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         LeaderNeverClearsGroupOwnedMonotonicPublications)
    {
        ProtocolFixture fixture(2u);
        const auto first = prepareCommand(
            fixture,
            MoEOverlayDeviceControllerTransactionKind::StaticCheck,
            0u,
            0u);
        admit(fixture, first);
        const auto first_group_records = fixture.groups;

        const auto second = fixture.protocol.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::StaticCheck,
            MoEOverlayDeviceDemandPhase::Invalid);
        ASSERT_TRUE(second.has_value());
        ASSERT_GT(*second, first);
        for (std::size_t group = 0u; group < fixture.groups.size(); ++group)
        {
            EXPECT_EQ(
                fixture.groups[group].snapshot_transaction,
                first_group_records[group].snapshot_transaction);
            EXPECT_EQ(
                fixture.groups[group].prepared_transaction,
                first_group_records[group].prepared_transaction);
            EXPECT_EQ(
                fixture.groups[group].published_transaction,
                first_group_records[group].published_transaction);
            EXPECT_TRUE(fixture.protocol.publishGroupSnapshot(
                static_cast<std::uint32_t>(group),
                *second,
                0x700u + group,
                64u + group));
        }
        EXPECT_TRUE(fixture.protocol.publishCommand(
            *second, 0u, 0x888u, 0u, 0u, 0u, 0u));
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         LLEPUsesTransientAssignmentAndRestoresEveryGroup)
    {
        ProtocolFixture fixture(4u);
        const auto transaction = prepareCommand(
            fixture,
            MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP,
            6u,
            0u);
        admit(fixture, transaction);

        EXPECT_EQ(fixture.protocol.currentDurableEpoch(), 7u);
        EXPECT_EQ(fixture.header.admission_epoch, 7u);
        EXPECT_EQ(fixture.protocol.activeLLEPTransaction(), transaction);
        ASSERT_TRUE(fixture.protocol.beginLLEPRestore(transaction));
        publishReverse(
            fixture.groups.size(),
            [&](std::uint32_t group)
            {
                return fixture.protocol.acknowledgeLLEPRestored(
                    group, transaction);
            });
        ASSERT_TRUE(fixture.protocol.completeLLEPRestore(transaction));
        EXPECT_EQ(fixture.protocol.activeLLEPTransaction(), 0u);
        EXPECT_EQ(fixture.protocol.currentDurableEpoch(), 7u);
        EXPECT_EQ(fixture.protocol.state(),
                  MoEOverlayDeviceControllerState::Complete);
        EXPECT_EQ(fixture.header.completed_transaction, transaction);
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         ConcurrentGroupPublicationUsesDisjointReleaseLanes)
    {
        ProtocolFixture fixture(7u);
        const auto transaction = fixture.protocol.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            MoEOverlayDeviceDemandPhase::Decode);
        ASSERT_TRUE(transaction.has_value());

        std::vector<std::thread> publishers;
        publishers.reserve(fixture.groups.size());
        std::array<bool, 7u> results{};
        for (std::uint32_t group = 0u; group < fixture.groups.size(); ++group)
        {
            publishers.emplace_back(
                [&, group]
                {
                    results[group] = fixture.protocol.publishGroupSnapshot(
                        group,
                        *transaction,
                        0x9000u + group,
                        1000u + group);
                });
        }
        for (auto &publisher : publishers)
            publisher.join();
        for (const bool result : results)
            EXPECT_TRUE(result);
        EXPECT_TRUE(fixture.protocol.allSnapshotsReady(*transaction));
    }

    TEST(Test__MoEOverlayDeviceControllerProtocol,
         StaleOrUnknownGroupPublicationPoisonsTheSharedAuthority)
    {
        ProtocolFixture fixture(2u);
        const auto transaction = fixture.protocol.beginTransaction(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            MoEOverlayDeviceDemandPhase::Decode);
        ASSERT_TRUE(transaction.has_value());

        EXPECT_FALSE(fixture.protocol.publishGroupSnapshot(
            4u, *transaction, 0x44u, 1u));
        EXPECT_EQ(fixture.protocol.state(),
                  MoEOverlayDeviceControllerState::Error);
        EXPECT_EQ(fixture.protocol.error(),
                  MoEOverlayDeviceControllerError::InvalidGroup);
        EXPECT_EQ(fixture.header.error_group_id, 4u);
    }
} // namespace llaminar2::test
