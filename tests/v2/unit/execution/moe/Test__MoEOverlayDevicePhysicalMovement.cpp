/**
 * @file Test__MoEOverlayDevicePhysicalMovement.cpp
 * @brief Device-free proof of command-authenticated physical MoE movement.
 *
 * Also verifies that fatal physical-progress diagnostics preserve immutable
 * endpoint identity, distinguish projection roles, and never mutate receipts.
 */

#include <gtest/gtest.h>

#include "execution/moe/MoEOverlayDeviceControllerKernels.h"
#include "execution/moe/MoEOverlayDevicePhysicalMovement.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr std::uint64_t kTopologyFingerprint =
            0xc894a5ef83d27109ULL;

        /** Build one exact participant without assigning meaning to its tier. */
        MoEExpertOwnerParticipant participant(
            int id,
            int tier,
            std::string tier_name,
            std::string domain,
            int domain_ordinal,
            int world_rank,
            GlobalDeviceAddress address)
        {
            return {
                .participant_id = id,
                .tier_idx = tier,
                .tier_name = std::move(tier_name),
                .domain_name = std::move(domain),
                .domain_participant_index = domain_ordinal,
                .address = address,
                .device = address.toLocalDeviceId(),
                .world_rank = world_rank,
                .world_rank_known = true,
            };
        }

        /**
         * Build a reversed-priority mixed-vendor topology: ROCm owns the sole
         * continuation authority and has the numerically smaller priority.
         */
        MoEOverlayDeviceControllerTopology topology()
        {
            MoEOverlayDeviceControllerTopology result;
            result.leader_participant_id = 2;
            result.leader_group_id = 1;
            result.leader_world_rank = 0;
            result.leader_device = DeviceId::rocm(0);
            result.participants = {
                participant(
                    0, 0, "integer_35", "cuda_capacity", 0, 1,
                    GlobalDeviceAddress::cuda(0, 1)),
                participant(
                    1, 0, "integer_35", "cuda_capacity", 1, 1,
                    GlobalDeviceAddress::cuda(1, 1)),
                participant(
                    2, 1, "integer_minus_12", "rocm_continuation", 0, 0,
                    GlobalDeviceAddress::rocm(0, 0)),
                participant(
                    3, 1, "integer_minus_12", "rocm_continuation", 1, 0,
                    GlobalDeviceAddress::rocm(1, 0)),
            };
            result.groups = {
                {
                    .group_id = 0,
                    .tier_index = 0,
                    .tier_priority = 35,
                    .domain_ordinal = 0,
                    .domain_name = "cuda_capacity",
                    .root_participant_id = 0,
                    .root_world_rank = 1,
                    .root_device = DeviceId::cuda(0),
                    .participant_ids = {0, 1},
                    .intra_group_transport =
                        MoEOverlayDeviceControllerIntraGroupTransport::
                            NativeCollective,
                    .inter_group_transport =
                        MoEOverlayDeviceControllerInterGroupTransport::
                            NodeLocalMapped,
                },
                {
                    .group_id = 1,
                    .tier_index = 1,
                    .tier_priority = -12,
                    .domain_ordinal = 1,
                    .domain_name = "rocm_continuation",
                    .root_participant_id = 2,
                    .root_world_rank = 0,
                    .root_device = DeviceId::rocm(0),
                    .participant_ids = {2, 3},
                    .intra_group_transport =
                        MoEOverlayDeviceControllerIntraGroupTransport::
                            NativeCollective,
                    .inter_group_transport =
                        MoEOverlayDeviceControllerInterGroupTransport::
                            LeaderLocal,
                },
            };
            result.topology_fingerprint = kTopologyFingerprint;
            return result;
        }

        /** Compute the exact device/host command digest for arbitrary entries. */
        std::uint64_t commandDigest(
            const std::vector<MoEOverlayDeviceMovementCommand> &entries)
        {
            std::uint64_t digest = moeOverlayCommandDigestSeed(
                static_cast<std::uint32_t>(entries.size()));
            constexpr std::size_t kWords =
                sizeof(MoEOverlayDeviceMovementCommand) /
                sizeof(std::uint64_t);
            const auto *words = reinterpret_cast<const std::uint64_t *>(
                entries.data());
            for (std::size_t index = 0u;
                 index < entries.size() * kWords;
                 ++index)
            {
                digest ^= moeOverlayCommandDigestWord(words[index], index);
            }
            return digest;
        }

        /** Finalize canonical header identity after entries are populated. */
        MoEOverlayDeviceTransportCommandBatch commandBatch(
            MoEOverlayDeviceControllerTransactionKind kind,
            std::uint64_t base_epoch,
            std::vector<MoEOverlayDeviceMovementCommand> entries,
            std::uint64_t topology_fingerprint = kTopologyFingerprint)
        {
            const bool durable_placement =
                kind == MoEOverlayDeviceControllerTransactionKind::
                            DynamicPlacement ||
                kind == MoEOverlayDeviceControllerTransactionKind::
                            PreparedContextRestore;
            const auto frozen_topology = topology();
            const auto participantPriority =
                [&frozen_topology](std::uint32_t participant_id)
            {
                const auto participant = std::find_if(
                    frozen_topology.participants.begin(),
                    frozen_topology.participants.end(),
                    [participant_id](const auto &candidate)
                    {
                        return candidate.participant_id ==
                               static_cast<int>(participant_id);
                    });
                if (participant == frozen_topology.participants.end())
                    throw std::logic_error(
                        "test command names an unknown participant");
                const auto group = std::find_if(
                    frozen_topology.groups.begin(),
                    frozen_topology.groups.end(),
                    [&participant](const auto &candidate)
                    {
                        return candidate.tier_index == participant->tier_idx;
                    });
                if (group == frozen_topology.groups.end())
                    throw std::logic_error(
                        "test participant has no immutable tier group");
                return group->tier_priority;
            };
            std::uint64_t bytes = 0u;
            std::vector<std::uint8_t> changed_layers(3u, 0u);
            std::uint32_t promotions = 0u;
            std::uint32_t demotions = 0u;
            std::uint32_t same_priority_moves = 0u;
            for (const auto &entry : entries)
            {
                bytes += entry.payload_bytes;
                if (!durable_placement)
                {
                    continue;
                }
                changed_layers.at(entry.layer) = 1u;
                const int source_priority =
                    participantPriority(entry.source_participant);
                const int destination_priority =
                    participantPriority(entry.destination_participant);
                if (destination_priority < source_priority)
                    ++promotions;
                else if (destination_priority > source_priority)
                    ++demotions;
                else
                    ++same_priority_moves;
            }
            const std::uint64_t candidate_epoch =
                durable_placement && !entries.empty()
                    ? base_epoch + 1u
                    : base_epoch;
            for (std::size_t index = 0u; index < entries.size(); ++index)
            {
                entries[index].ordinal = static_cast<std::uint32_t>(index);
                if (entries[index].op !=
                    static_cast<std::uint32_t>(
                        MoEOverlayDeviceMovementOp::TransientAssignment))
                {
                    entries[index].payload_slot =
                        static_cast<std::uint32_t>(index);
                }
                entries[index].source_epoch = base_epoch;
                entries[index].candidate_epoch = candidate_epoch;
            }
            MoEOverlayDeviceTransportCommandBatch result;
            result.header.kind = static_cast<std::uint32_t>(kind);
            result.header.demand_phase = static_cast<std::uint32_t>(
                kind == MoEOverlayDeviceControllerTransactionKind::
                            DynamicPlacement
                    ? MoEOverlayDeviceDemandPhase::Decode
                    : MoEOverlayDeviceDemandPhase::Invalid);
            result.header.command_count =
                static_cast<std::uint32_t>(entries.size());
            result.header.topology_fingerprint = topology_fingerprint;
            result.header.transaction_id = 91u;
            result.header.base_epoch = base_epoch;
            result.header.candidate_epoch = candidate_epoch;
            result.header.command_digest = commandDigest(entries);
            result.header.packed_weight_bytes = bytes;
            result.header.parallel_command_count = result.header.command_count;
            result.header.movement_round_count =
                entries.empty() ? 0u : 1u;
            if (durable_placement && !entries.empty())
            {
                result.header.snapshot_observations = entries.size();
                result.header.accepted_cycles = 1u;
                result.header.promotions = promotions;
                result.header.demotions = demotions;
                result.header.same_priority_moves = same_priority_moves;
                result.header.changed_layers = static_cast<std::uint32_t>(
                    std::count(changed_layers.begin(),
                               changed_layers.end(),
                               std::uint8_t{1u}));
                if (kind == MoEOverlayDeviceControllerTransactionKind::
                                DynamicPlacement)
                {
                    result.header.projected_service_gain_ns = 1'000u;
                    result.header.projected_transfer_and_repack_ns = 100u;
                    result.header.projected_inference_interference_ns = 100u;
                    result.header.projected_net_benefit_ns = 800u;
                    if (promotions != 0u || demotions != 0u)
                    {
                        result.header.priority_cost_before = 2u;
                        result.header.priority_cost_after = 1u;
                    }
                    else
                    {
                        result.header.same_priority_makespan_before = 2u;
                        result.header.same_priority_makespan_after = 1u;
                    }
                }
            }
            result.entries = std::move(entries);
            result.participant_count = 4u;
            result.num_layers = 3u;
            result.num_experts = 16u;
            return result;
        }

        /** Construct one physical movement entry before canonical ordinals. */
        MoEOverlayDeviceMovementCommand move(
            MoEOverlayDeviceMovementOp op,
            std::uint32_t layer,
            std::uint32_t expert,
            std::uint32_t source,
            std::uint32_t destination,
            std::uint64_t bytes = 4096u,
            MoEOverlayDeviceMovementAxis axis =
                MoEOverlayDeviceMovementAxis::TierResidency)
        {
            return {
                .op = static_cast<std::uint32_t>(op),
                .layer = layer,
                .expert = expert,
                .source_participant = source,
                .destination_participant = destination,
                .flags = static_cast<std::uint32_t>(axis),
                .payload_bytes = bytes,
            };
        }

        /** Construct one resident-only LLEP assignment entry. */
        MoEOverlayDeviceMovementCommand assignment(
            std::uint32_t layer,
            std::uint32_t expert,
            std::uint32_t participant)
        {
            return {
                .op = static_cast<std::uint32_t>(
                    MoEOverlayDeviceMovementOp::TransientAssignment),
                .layer = layer,
                .expert = expert,
                .source_participant = participant,
                .destination_participant = participant,
                .payload_slot = kMoEOverlayDeviceInvalidSlot,
                .flags = static_cast<std::uint32_t>(
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
            };
        }
    } // namespace

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         ProjectionDiagnosticsPreserveExactPendingEndpointsAndReadiness)
    {
        MoEOverlayDevicePhysicalMovementBatch batch;
        batch.transaction_id = 36u;
        batch.packed_weight_bytes = 8192u;
        batch.migrations.resize(2);
        for (auto &migration : batch.migrations)
        {
            migration.layer_idx = 47;
            migration.expert_id = 9;
            migration.source.owner_world_rank = 1;
            migration.source.owner_participant = 3;
            migration.source.device = DeviceId::rocm(1);
            migration.destination.owner_world_rank = 0;
            migration.destination.owner_participant = 0;
            migration.destination.device = DeviceId::cuda(0);
        }
        const std::array<std::uint8_t, 6> ready{1, 0, 1, 0, 1, 0};
        const auto description = batch.describeProjectionReadiness(ready);
        EXPECT_NE(description.find("transaction=36,payload_bytes=8192,projections_ready=3/6"), std::string::npos);
        EXPECT_NE(description.find("migration=0,layer=47,expert=9,projection=up"), std::string::npos);
        EXPECT_EQ(description.find("migration=0,layer=47,expert=9,projection=gate"), std::string::npos);
        EXPECT_NE(description.find("migration=1,layer=47,expert=9,projection=gate"), std::string::npos);
        EXPECT_NE(description.find("migration=1,layer=47,expert=9,projection=down"), std::string::npos);
        EXPECT_NE(description.find("source=1:" + DeviceId::rocm(1).toString() + ":participant=3"), std::string::npos);
        EXPECT_NE(description.find("destination=0:" + DeviceId::cuda(0).toString() + ":participant=0"), std::string::npos);
        EXPECT_EQ(ready, (std::array<std::uint8_t, 6>{1, 0, 1, 0, 1, 0}));
        const std::array<std::uint8_t, 6> complete{1, 1, 1, 1, 1, 1};
        EXPECT_NE(batch.describeProjectionReadiness(complete).find("projections_ready=6/6,pending=[]"), std::string::npos);
        EXPECT_THROW(batch.describeProjectionReadiness(std::span(ready).first(5)), std::invalid_argument);
        EXPECT_THROW(batch.describeProjectionReadiness({}), std::invalid_argument);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         DurableCommandsBecomeOneClosedCycleInEndpointOrder)
    {
        const auto frozen_topology = topology();
        ASSERT_TRUE(frozen_topology.valid());

        // Entries are destination-canonical, deliberately not cycle ordered.
        const auto command = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            7u,
            {
                move(MoEOverlayDeviceMovementOp::DurableMove, 1, 3, 3, 0),
                move(MoEOverlayDeviceMovementOp::DurableMove, 1, 2, 2, 1),
                move(MoEOverlayDeviceMovementOp::DurableMove, 1, 0, 0, 2),
                move(MoEOverlayDeviceMovementOp::DurableMove, 1, 1, 1, 3),
            });
        ASSERT_TRUE(command.valid());

        const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(
            command, frozen_topology);
        ASSERT_TRUE(physical.valid());
        ASSERT_TRUE(physical.movesWeights());
        ASSERT_EQ(physical.migrations.size(), 4u);
        ASSERT_EQ(physical.migration_cycles.size(), 1u);
        EXPECT_TRUE(physical.migration_cycles.front().valid(
            physical.migrations));
        EXPECT_EQ(physical.shadow_requirements.size(), 4u);
        EXPECT_EQ(physical.packed_weight_bytes, 4u * 4096u);

        // Smaller integer priority is the only promotion rule; ROCm and the
        // continuation role do not receive a hard-coded preference.
        EXPECT_EQ(
            physical.migrations[0].direction,
            MoEOverlayTierMigrationDirection::Demotion);
        EXPECT_EQ(
            physical.migrations[2].direction,
            MoEOverlayTierMigrationDirection::Promotion);
        EXPECT_TRUE(physical.migrations[0].source.device.is_rocm());
        EXPECT_TRUE(physical.migrations[0].destination.device.is_cuda());
        EXPECT_EQ(physical.migrations[0].source.owner_world_rank, 0);
        EXPECT_EQ(physical.migrations[0].destination.owner_world_rank, 1);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         BoundedPollCursorIsFairAndNeverRepeatsWithinAQuantum)
    {
        MoEOverlayPhysicalWavePollCursor cursor(
            /*operation_count=*/8u,
            /*maximum_polls_per_quantum=*/3u);
        std::vector<std::uint8_t> ready(8u, 0u);

        cursor.beginQuantum();
        EXPECT_EQ(cursor.nextPending(ready), 0u);
        EXPECT_EQ(cursor.nextPending(ready), 1u);
        EXPECT_EQ(cursor.nextPending(ready), 2u);

        // A fresh quantum resumes after the prior selection rather than
        // repeatedly favoring the first unfinished operation.
        cursor.beginQuantum();
        EXPECT_EQ(cursor.nextPending(ready), 3u);
        EXPECT_EQ(cursor.nextPending(ready), 4u);
        EXPECT_EQ(cursor.nextPending(ready), 5u);

        ready[6] = 1u;
        cursor.beginQuantum();
        EXPECT_EQ(cursor.nextPending(ready), 7u);
        EXPECT_EQ(cursor.nextPending(ready), 0u);
        EXPECT_EQ(cursor.nextPending(ready), 1u);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         BoundedPollCursorSkipsReadyWorkAndEndsAfterOneCompleteScan)
    {
        MoEOverlayPhysicalWavePollCursor cursor(
            /*operation_count=*/5u,
            /*maximum_polls_per_quantum=*/5u);
        std::vector<std::uint8_t> ready{1u, 0u, 1u, 0u, 1u};

        cursor.beginQuantum();
        EXPECT_EQ(cursor.nextPending(ready), 1u);
        EXPECT_EQ(cursor.nextPending(ready), 3u);
        EXPECT_FALSE(cursor.nextPending(ready).has_value());

        EXPECT_THROW(
            (MoEOverlayPhysicalWavePollCursor(0u, 1u)),
            std::invalid_argument);
        EXPECT_THROW(
            (MoEOverlayPhysicalWavePollCursor(1u, 0u)),
            std::invalid_argument);
        EXPECT_THROW(
            static_cast<void>(
                cursor.nextPending(std::span<const std::uint8_t>{})),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         SamePriorityRebalanceRetainsItsOwnMovementAxis)
    {
        const auto command = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            12u,
            {
                move(
                    MoEOverlayDeviceMovementOp::DurableMove,
                    0,
                    4,
                    1,
                    0,
                    4096u,
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
                move(
                    MoEOverlayDeviceMovementOp::DurableMove,
                    0,
                    5,
                    0,
                    1,
                    4096u,
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
            });
        ASSERT_TRUE(command.valid());

        const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(
            command, topology());
        ASSERT_TRUE(physical.valid());
        ASSERT_EQ(physical.migrations.size(), 2u);
        EXPECT_EQ(
            physical.migrations[0].direction,
            MoEOverlayTierMigrationDirection::SamePriority);
        EXPECT_EQ(
            physical.migrations[1].direction,
            MoEOverlayTierMigrationDirection::SamePriority);
        EXPECT_TRUE(std::all_of(
            physical.migrations.begin(),
            physical.migrations.end(),
            [](const auto &migration)
            {
                return migration.axis ==
                       MoEOptimizationMovementAxis::ParticipantPlacement;
            }));
        ASSERT_EQ(physical.migration_cycles.size(), 1u);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         UnbalancedDynamicEdgesFailBeforeAnyPhysicalSlotCanBeReserved)
    {
        const auto command = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            4u,
            {move(MoEOverlayDeviceMovementOp::DurableMove, 0, 1, 0, 2)});
        ASSERT_TRUE(command.valid())
            << "the transport ABI alone intentionally cannot infer capacities";
        EXPECT_THROW(
            (void)makeMoEOverlayDevicePhysicalMovementBatch(
                command, topology()),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         TopologyMismatchAndTamperedBytesCannotReachPhysicalPreparation)
    {
        auto command = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            8u,
            {
                move(MoEOverlayDeviceMovementOp::DurableMove, 0, 1, 1, 0),
                move(MoEOverlayDeviceMovementOp::DurableMove, 0, 2, 0, 1),
            });
        ASSERT_TRUE(command.valid());

        auto wrong_topology = topology();
        wrong_topology.topology_fingerprint ^= 0x1u;
        ASSERT_TRUE(wrong_topology.valid());
        EXPECT_THROW(
            (void)makeMoEOverlayDevicePhysicalMovementBatch(
                command, wrong_topology),
            std::invalid_argument);

        command.entries[0].payload_bytes += 1u;
        ASSERT_FALSE(command.valid());
        EXPECT_THROW(
            (void)makeMoEOverlayDevicePhysicalMovementBatch(
                command, topology()),
            std::invalid_argument);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         LLEPSeparatesTransientArrivalsFromResidentAssignments)
    {
        const auto command = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP,
            17u,
            {
                assignment(2, 5, 1),
                move(
                    MoEOverlayDeviceMovementOp::TransientArrival,
                    2,
                    7,
                    0,
                    2,
                    8192u,
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
            });
        ASSERT_TRUE(command.valid());

        const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(
            command, topology());
        ASSERT_TRUE(physical.valid());
        ASSERT_EQ(physical.migrations.size(), 1u);
        EXPECT_TRUE(physical.migration_cycles.empty());
        ASSERT_EQ(physical.shadow_requirements.size(), 1u);
        EXPECT_EQ(physical.command_count, 2u);
        EXPECT_EQ(physical.packed_weight_bytes, 8192u);
        EXPECT_EQ(
            physical.migrations.front().direction,
            MoEOverlayTierMigrationDirection::Promotion);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         NoOpDynamicTransactionProducesValidZeroMovementEvidence)
    {
        const auto command = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
            23u,
            {});
        ASSERT_TRUE(command.valid());

        const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(
            command, topology());
        EXPECT_TRUE(physical.valid());
        EXPECT_FALSE(physical.movesWeights());
        EXPECT_TRUE(physical.migrations.empty());
        EXPECT_TRUE(physical.migration_cycles.empty());
        EXPECT_TRUE(physical.shadow_requirements.empty());
        EXPECT_EQ(physical.base_epoch, physical.candidate_epoch);
    }

    TEST(Test__MoEOverlayDevicePhysicalMovement,
         PreparedContextRestoreUsesTheSameCapacityPreservingDurablePipeline)
    {
        const auto movement = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::
                PreparedContextRestore,
            31u,
            {
                move(
                    MoEOverlayDeviceMovementOp::DurableMove,
                    0,
                    4,
                    1,
                    0,
                    4096u,
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
                move(
                    MoEOverlayDeviceMovementOp::DurableMove,
                    0,
                    5,
                    0,
                    1,
                    4096u,
                    MoEOverlayDeviceMovementAxis::ParticipantPlacement),
            });
        ASSERT_TRUE(movement.valid());

        const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(
            movement, topology());
        ASSERT_TRUE(physical.valid());
        ASSERT_TRUE(physical.movesWeights());
        ASSERT_EQ(physical.migration_cycles.size(), 1u);
        EXPECT_EQ(
            physical.kind,
            MoEOverlayDeviceControllerTransactionKind::
                PreparedContextRestore);

        const auto certification = commandBatch(
            MoEOverlayDeviceControllerTransactionKind::
                PreparedContextRestore,
            physical.candidate_epoch,
            {});
        ASSERT_TRUE(certification.valid());
        const auto terminal = makeMoEOverlayDevicePhysicalMovementBatch(
            certification, topology());
        EXPECT_TRUE(terminal.valid());
        EXPECT_FALSE(terminal.movesWeights());
        EXPECT_EQ(terminal.base_epoch, terminal.candidate_epoch);
    }
} // namespace llaminar2::test
