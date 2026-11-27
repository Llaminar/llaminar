/**
 * @file Test__MoEOverlayDeviceControllerProtocol.cpp
 * @brief Adversarial CPU tests for the shared GPU controller lifecycle ABI.
 *
 * These tests execute the same acquire/release state machine that CUDA and HIP
 * controller kernels consume. They deliberately use a variable number of
 * independently publishing groups and must remain device-free and fast.
 */

#include "execution/moe/MoEOverlayDeviceControllerProtocol.h"
#include "execution/moe/MoEOverlayDeviceControllerDispatch.h"
#include "execution/moe/MoEOverlayCycleSearch.h"
#include "execution/moe/MoEOverlayWorkerDrainProtocol.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <thread>
#include <vector>

namespace llaminar2::test
{

    /** Each graph action selects itself exactly once; no runtime policy runs here. */
    TEST(Test__MoEOverlayDeviceControllerProtocol, CapturedActionDispatchIsExact)
    {
        using Action = MoEOverlayDeviceControllerAction;
        constexpr std::array actions{
            Action::BeginTransaction,
            Action::PublishGroupSnapshot,
            Action::PublishCommand,
            Action::AcknowledgePrepared,
            Action::BeginCommit,
            Action::AcknowledgePublished,
            Action::PublishAdmission,
            Action::BeginDynamicRetirement,
            Action::AcknowledgeRetired,
            Action::CompleteDynamicRetirement,
            Action::BeginLLEPRestore,
            Action::AcknowledgeLLEPRestored,
            Action::CompleteLLEPRestore,
            Action::AuthorStaticPolicy,
            Action::AwaitTransactionComplete,
            Action::AuthorDynamicPolicy,
            Action::PublishParticipantSnapshot,
            Action::ApplyRuntimeCandidate,
            Action::PublishRuntimeCandidate,
            Action::PublishRuntimeRetirement,
            Action::AwaitRuntimeCommit,
            Action::AwaitRuntimeRetirement,
            Action::PublishRuntimeRetirementReadiness,
            Action::CompleteEmptyDynamicDecision,
            Action::AuthorPreparedContextRestore,
        };
        for (const auto expected : actions)
        {
            int calls = 0;
            EXPECT_TRUE(dispatchMoEOverlayControllerAction(
                expected, [&]<Action Actual>()
                {
                    ++calls;
                    EXPECT_EQ(Actual, expected);
                    return true;
                }));
            EXPECT_EQ(calls, 1);
            EXPECT_FALSE(dispatchMoEOverlayControllerAction(
                expected, []<Action>() { return false; }));
        }
        for (const auto invalid : {Action::Invalid, static_cast<Action>(0xffffffffu)})
        {
            int calls = 0;
            EXPECT_FALSE(dispatchMoEOverlayControllerAction(
                invalid, [&]<Action>() { ++calls; return true; }));
            EXPECT_EQ(calls, 0);
        }
    }
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
         PreparedContextRestoreAdvancesThroughTheDurableRetirementStateMachine)
    {
        ProtocolFixture fixture(3u);
        const auto transaction = prepareCommand(
            fixture,
            MoEOverlayDeviceControllerTransactionKind::
                PreparedContextRestore,
            2u,
            8u * 1024u * 1024u);

        admit(fixture, transaction);
        EXPECT_EQ(fixture.protocol.currentDurableEpoch(), 8u);
        EXPECT_EQ(fixture.header.admission_epoch, 8u);
        ASSERT_TRUE(fixture.protocol.beginDynamicRetirement(transaction));
        publishReverse(
            fixture.groups.size(),
            [&](std::uint32_t group)
            {
                return fixture.protocol.acknowledgeRetired(
                    group, transaction, 7u);
            });
        ASSERT_TRUE(
            fixture.protocol.completeDynamicRetirement(transaction));
        EXPECT_EQ(
            fixture.protocol.state(),
            MoEOverlayDeviceControllerState::Complete);
        EXPECT_EQ(fixture.header.completed_transaction, transaction);
        EXPECT_EQ(
            fixture.protocol.error(),
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

    /**
     * @brief Replay compressed minimum-edge cycle search + first_start cursor
     *        against full findMoEOverlayPlacementCycle across dense cycles and
     *        dead-end DAG components for 5-target and 8-target topologies.
     */
    TEST(Test__MoEOverlayDeviceControllerProtocol,
         CycleSearchFirstStartAndCompressedGraphMatchOracleReplay)
    {
        constexpr std::uint16_t kNone = 0xffffu;
        for (const std::uint32_t target_count : {5u, 8u})
        {
            constexpr std::uint32_t expert_count = 512u;
            std::vector<std::int32_t> current(expert_count);
            std::vector<std::int32_t> desired(expert_count);
            // Deterministic unbalanced+balanced mix that produces both rich
            // multi-length cycles and large dead-end DAG remainders after
            // partial enumeration.
            for (std::uint32_t e = 0u; e < expert_count; ++e)
            {
                current[e] = static_cast<std::int32_t>((e * 7u + 3u) % target_count);
                desired[e] = static_cast<std::int32_t>(
                    (e < 320u) ? ((e * 11u + 1u) % target_count)
                               : ((current[e] + 1u) % (target_count - 1u)));
            }

            std::vector<std::uint8_t> enum_oracle(expert_count, 0u);
            std::vector<std::uint8_t> enum_cursor(expert_count, 0u);
            std::vector<std::uint8_t> enum_compressed(expert_count, 0u);

            std::vector<std::uint8_t> visited(target_count, 0u);
            std::vector<std::uint32_t> sources(target_count, 0u);
            std::vector<std::uint32_t> next_edges(target_count, 0u);
            std::vector<std::uint32_t> cycle_oracle(target_count, 0u);
            std::vector<std::uint32_t> cycle_cursor(target_count, 0u);
            std::vector<std::uint32_t> cycle_compressed(target_count, 0u);

            std::vector<std::uint16_t> pair_head(target_count * target_count, kNone);
            std::vector<std::uint16_t> pair_next(expert_count, kNone);
            std::vector<std::uint8_t> dead_pair(target_count * target_count, 0u);
            for (std::uint32_t rev = expert_count; rev != 0u; --rev)
            {
                const std::uint32_t e = rev - 1u;
                if (current[e] == desired[e])
                    continue;
                const std::uint32_t slot =
                    static_cast<std::uint32_t>(current[e]) * target_count +
                    static_cast<std::uint32_t>(desired[e]);
                pair_next[e] = pair_head[slot];
                pair_head[slot] = static_cast<std::uint16_t>(e);
            }

            std::uint32_t search_first_cursor = 0u;
            std::uint32_t search_first_compressed = 0u;
            std::uint32_t cycles_found = 0u;

            while (true)
            {
                const auto oracle_res = findMoEOverlayPlacementCycle(
                    current.data(), desired.data(), enum_oracle,
                    target_count, expert_count, visited.data(),
                    sources.data(), next_edges.data(), cycle_oracle.data(), 0u);
                const auto cursor_res = findMoEOverlayPlacementCycle(
                    current.data(), desired.data(), enum_cursor,
                    target_count, expert_count, visited.data(),
                    sources.data(), next_edges.data(), cycle_cursor.data(),
                    search_first_cursor);

                // Compressed canonical DFS matching authorDynamicPolicy.
                std::uint32_t comp_len = 0u;
                std::vector<std::uint8_t> dfs_vis(target_count, 0u);
                std::vector<std::uint8_t> dfs_src(target_count, 0u);
                std::vector<std::uint16_t> dfs_min(target_count, 0u);
                for (std::uint32_t first = search_first_compressed;
                     first < expert_count;
                     ++first)
                {
                    if (enum_compressed[first] != 0u)
                        continue;
                    const auto start = static_cast<std::uint32_t>(current[first]);
                    const auto next = static_cast<std::uint32_t>(desired[first]);
                    if (start == next || dead_pair[start * target_count + next] != 0u)
                        continue;
                    cycle_compressed[0] = first;
                    std::fill(dfs_vis.begin(), dfs_vis.end(), 0u);
                    dfs_vis[start] = 1u;
                    dfs_vis[next] = 1u;
                    std::uint32_t depth = 1u;
                    dfs_src[1] = static_cast<std::uint8_t>(next);
                    dfs_min[1] = 0u;
                    while (true)
                    {
                        const std::uint32_t u = dfs_src[depth];
                        const std::uint16_t min_e = dfs_min[depth];
                        std::uint16_t best_e = kNone;
                        std::uint32_t best_v = target_count;
                        const std::uint32_t row_base = u * target_count;
                        for (std::uint32_t v = 0u; v < target_count; ++v)
                        {
                            if (v == u)
                                continue;
                            std::uint16_t h = pair_head[row_base + v];
                            while (h != kNone && enum_compressed[h] != 0u)
                                h = pair_next[h];
                            pair_head[row_base + v] = h;
                            if (h != kNone && h >= min_e && h < best_e)
                            {
                                best_e = h;
                                best_v = v;
                            }
                        }
                        if (best_e == kNone)
                        {
                            if (depth == 1u)
                                break;
                            --depth;
                            continue;
                        }
                        dfs_min[depth] = static_cast<std::uint16_t>(best_e + 1u);
                        if (best_v == start)
                        {
                            cycle_compressed[depth] = best_e;
                            comp_len = depth + 1u;
                            break;
                        }
                        if (dfs_vis[best_v] == 0u && depth + 1u < target_count)
                        {
                            dfs_vis[best_v] = 1u;
                            cycle_compressed[depth] = best_e;
                            ++depth;
                            dfs_src[depth] = static_cast<std::uint8_t>(best_v);
                            dfs_min[depth] = 0u;
                        }
                    }
                    if (comp_len != 0u)
                        break;
                    dead_pair[start * target_count + next] = 1u;
                }

                ASSERT_EQ(cursor_res.length, oracle_res.length)
                    << "target_count=" << target_count << " cycle=" << cycles_found;
                ASSERT_EQ(comp_len, oracle_res.length)
                    << "target_count=" << target_count << " cycle=" << cycles_found;
                if (oracle_res.length == 0u)
                    break;

                for (std::uint32_t i = 0u; i < oracle_res.length; ++i)
                {
                    ASSERT_EQ(cycle_cursor[i], cycle_oracle[i]);
                    ASSERT_EQ(cycle_compressed[i], cycle_oracle[i]);
                    enum_oracle[cycle_oracle[i]] = 1u;
                    enum_cursor[cycle_cursor[i]] = 1u;
                    enum_compressed[cycle_compressed[i]] = 1u;
                }
                search_first_cursor = cycle_cursor[0] + 1u;
                search_first_compressed = cycle_compressed[0] + 1u;
                ++cycles_found;
            }
            EXPECT_GT(cycles_found, 30u);
        }
    }
} // namespace llaminar2::test
