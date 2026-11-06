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
#include "execution/moe/MoEOverlayCycleSearch.h"
#include "execution/moe/MoEOverlayPhysicalLaneDemand.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <tuple>
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
            std::uint32_t layer_count = 3u;
            for (const auto &entry : entries)
                layer_count = std::max(layer_count, entry.layer + 1u);
            std::vector<std::uint8_t> changed_layers(layer_count, 0u);
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
            result.num_layers = layer_count;
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
                .flags = moeOverlayMovementFlags(axis,
                    op == MoEOverlayDeviceMovementOp::DurableMove ? 1u : 0u),
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

    /** Both durable objectives translate every row without changing payload volume. */
    TEST(Test__MoEOverlayDevicePhysicalMovement,
         PipelineStageDurableCyclesRetainGlobalLayerIdentity)
    {
        for (const auto kind : {MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
                               MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore})
        for (const int origin : {0, 32, 40, std::numeric_limits<int>::max() - 3})
        for (std::uint32_t row = 0u; row < 3u; ++row)
        {
            SCOPED_TRACE(::testing::Message() << "kind=" << static_cast<int>(kind)
                << " origin=" << origin << " row=" << row);
            auto command = commandBatch(kind, 7u, {
                move(MoEOverlayDeviceMovementOp::DurableMove, row, 3, 3, 0),
                move(MoEOverlayDeviceMovementOp::DurableMove, row, 2, 2, 1),
                move(MoEOverlayDeviceMovementOp::DurableMove, row, 0, 0, 2),
                move(MoEOverlayDeviceMovementOp::DurableMove, row, 1, 1, 3),
            });
            const auto zero_origin = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
            command.first_model_layer = origin;
            ASSERT_TRUE(command.valid());
            const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
            ASSERT_TRUE(physical.valid());
            EXPECT_EQ(physical.first_model_layer, origin);
            EXPECT_EQ(physical.num_layers, 3u);
            EXPECT_EQ(physical.packed_weight_bytes, 16384u);
            ASSERT_EQ(physical.migrations.size(), 4u);
            ASSERT_EQ(physical.migration_cycles.size(), 1u);
            ASSERT_EQ(physical.shadow_requirements.size(), 4u);
            const int global_layer = origin + static_cast<int>(row);
            EXPECT_EQ(physical.migration_cycles.front().layer_idx, global_layer);
            for (std::size_t index = 0; index < physical.migrations.size(); ++index)
            {
                const auto &migration = physical.migrations[index];
                EXPECT_EQ(migration.layer_idx, global_layer);
                EXPECT_EQ(migration.source.layer_idx, global_layer);
                EXPECT_EQ(migration.destination.layer_idx, global_layer);
                EXPECT_EQ(migration.estimated_weight_bytes, 4096u);
                EXPECT_EQ(migration.source.device, zero_origin.migrations[index].source.device);
                EXPECT_EQ(migration.destination.device, zero_origin.migrations[index].destination.device);
                EXPECT_EQ(command.entries[index].layer, row);
                EXPECT_EQ(physical.shadow_requirements[index].layer_idx, global_layer);
                EXPECT_EQ(physical.shadow_requirements[index].slot_count, 1u);
            }
            EXPECT_EQ(physical.execution_fingerprint == zero_origin.execution_fingerprint, origin == 0);
        }
    }

    /** Transient assignments retain compact device rows; only actual arrivals move bytes. */
    TEST(Test__MoEOverlayDevicePhysicalMovement,
         PipelineStageLLEPArrivalsAndAssignmentsKeepExactExtent)
    {
        for (const int origin : {0, 32, 40, std::numeric_limits<int>::max() - 3})
        {
            auto command = commandBatch(MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP,
                17u, {assignment(2, 5, 1),
                      move(MoEOverlayDeviceMovementOp::TransientArrival, 2, 7, 0, 2,
                           8192u, MoEOverlayDeviceMovementAxis::ParticipantPlacement)});
            command.first_model_layer = origin;
            const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
            ASSERT_TRUE(physical.valid());
            ASSERT_EQ(physical.migrations.size(), 1u);
            EXPECT_EQ(physical.migrations[0].layer_idx, origin + 2);
            EXPECT_EQ(physical.migrations[0].source.layer_idx, origin + 2);
            EXPECT_EQ(physical.migrations[0].destination.layer_idx, origin + 2);
            EXPECT_EQ(physical.command_count, 2u);
            EXPECT_EQ(physical.packed_weight_bytes, 8192u);
            EXPECT_TRUE(physical.migration_cycles.empty());
            ASSERT_EQ(physical.shadow_requirements.size(), 1u);
            EXPECT_EQ(physical.shadow_requirements[0].layer_idx, origin + 2);

            auto resident = commandBatch(MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP,
                17u, {assignment(2, 5, 1)});
            resident.first_model_layer = origin;
            const auto assignment_only = makeMoEOverlayDevicePhysicalMovementBatch(resident, topology());
            ASSERT_TRUE(assignment_only.valid());
            EXPECT_EQ(assignment_only.first_model_layer, origin);
            EXPECT_FALSE(assignment_only.movesWeights());
            EXPECT_EQ(assignment_only.packed_weight_bytes, 0u);
        }
    }

    /** No-movement receipts still distinguish equal-shaped stages and bound arithmetic. */
    TEST(Test__MoEOverlayDevicePhysicalMovement,
         PipelineStageNoOpIdentityAndInvalidIntervals)
    {
        for (const auto kind : {MoEOverlayDeviceControllerTransactionKind::StaticCheck,
                               MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
                               MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore})
        {
            auto command = commandBatch(kind, 23u, {});
            const auto zero_origin = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
            command.first_model_layer = 32;
            const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
            ASSERT_TRUE(physical.valid());
            EXPECT_EQ(physical.first_model_layer, 32);
            EXPECT_FALSE(physical.movesWeights());
            EXPECT_NE(physical.execution_fingerprint, zero_origin.execution_fingerprint);
            for (const int invalid_origin : {-1, std::numeric_limits<int>::max() - 2})
            {
                command.first_model_layer = invalid_origin;
                EXPECT_FALSE(command.valid());
                EXPECT_THROW((void)makeMoEOverlayDevicePhysicalMovementBatch(command, topology()),
                    std::invalid_argument);
                auto invalid = physical;
                invalid.first_model_layer = invalid_origin;
                EXPECT_FALSE(invalid.valid());
            }
        }
    }

    /** A coherent foreign migration must fail before cycle or shadow-slot admission. */
    TEST(Test__MoEOverlayDevicePhysicalMovement,
         PipelineStageRejectsForeignPhysicalLayerCoordinates)
    {
        auto command = commandBatch(MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP,
            17u, {move(MoEOverlayDeviceMovementOp::TransientArrival, 1, 7, 0, 2,
                8192u, MoEOverlayDeviceMovementAxis::ParticipantPlacement)});
        command.first_model_layer = 32;
        const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
        ASSERT_TRUE(physical.valid());
        for (const int foreign_layer : {0, 31, 35, std::numeric_limits<int>::max()})
        {
            auto foreign = physical;
            foreign.migrations[0].layer_idx = foreign_layer;
            foreign.migrations[0].source.layer_idx = foreign_layer;
            foreign.migrations[0].destination.layer_idx = foreign_layer;
            foreign.shadow_requirements[0].layer_idx = foreign_layer;
            EXPECT_FALSE(foreign.valid()) << foreign_layer;
        }
    }

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

    /** @test Invalid objective/cycle inputs cannot alias an admitted wire identity. */
    TEST(Test__MoEOverlayDevicePhysicalMovement, MovementFlagsRejectInvalidIdentity)
    {
        constexpr auto maximum_cycle = UINT32_MAX >> 2u;
        for (const auto raw_axis : {0u, 4u, 5u, UINT32_MAX})
            for (const auto cycle : {0u, 1u, maximum_cycle})
            {
                const auto flags = moeOverlayMovementFlags(
                    static_cast<MoEOverlayDeviceMovementAxis>(raw_axis), cycle);
                EXPECT_EQ(flags, 0u);
                EXPECT_EQ(moeOverlayMovementCycle(flags), 0u);
                EXPECT_EQ(static_cast<std::uint32_t>(moeOverlayMovementAxis(flags)), 0u);
            }
        for (const auto axis : {MoEOverlayDeviceMovementAxis::TierResidency,
                                MoEOverlayDeviceMovementAxis::ParticipantPlacement,
                                MoEOverlayDeviceMovementAxis::Combined})
        {
            for (const auto cycle : {0u, 1u, maximum_cycle})
            {
                const auto flags = moeOverlayMovementFlags(axis, cycle);
                EXPECT_EQ(moeOverlayMovementAxis(flags), axis);
                EXPECT_EQ(moeOverlayMovementCycle(flags), cycle);
            }
            for (const auto overflow : {maximum_cycle + 1u, UINT32_MAX})
                EXPECT_EQ(moeOverlayMovementFlags(axis, overflow), 0u);
        }
    }

    /**
     * @test Original production cycle search survives canonical destination sort.
     * Two opposite authored triangles must consume two lanes, not the three
     * swaps found by regrouping; three later layers prove the default 5/6 case.
     */
    TEST(Test__MoEOverlayDevicePhysicalMovement,
         AuthoredCycleIdentityRetainsOriginalLaneBound)
    {
        for (const std::uint32_t admitted : {2u, 5u})
        {
            std::array<std::int32_t, 6> current{0, 1, 0, 2, 1, 2};
            const std::array<std::int32_t, 6> desired{1, 2, 2, 1, 0, 0};
            std::array<std::uint8_t, 6> excluded{};
            std::array<std::uint8_t, 4> visited{};
            std::array<std::uint32_t, 5> sources{}, next{}, cycle{};
            std::vector<MoEOverlayDeviceMovementCommand> entries;
            std::uint32_t authored = 0u;
            for (;;)
            {
                const auto found = findMoEOverlayPlacementCycle(
                    current.data(), desired.data(), excluded.data(), 4u, 6u,
                    visited.data(), sources.data(), next.data(), cycle.data());
                if (found.length == 0u)
                    break;
                ++authored;
                for (std::uint32_t index = 0u; index < found.length; ++index)
                {
                    const auto expert = cycle[index];
                    auto entry = move(MoEOverlayDeviceMovementOp::DurableMove,
                                      0u, expert, current[expert], desired[expert]);
                    entry.flags = moeOverlayMovementFlags(
                        MoEOverlayDeviceMovementAxis::TierResidency, authored);
                    entries.push_back(entry);
                    current[expert] = desired[expert];
                }
            }
            ASSERT_EQ(authored, 2u);
            for (std::uint32_t id = 3u; id <= admitted; ++id)
                for (std::uint32_t source : {0u, 1u})
                {
                    auto entry = move(MoEOverlayDeviceMovementOp::DurableMove,
                                      id - 2u, source, source, 1u - source);
                    entry.flags = moeOverlayMovementFlags(
                        MoEOverlayDeviceMovementAxis::TierResidency, id);
                    entries.push_back(entry);
                }
            std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b)
            {
                return std::tie(a.destination_participant, a.layer, a.expert) <
                       std::tie(b.destination_participant, b.layer, b.expert);
            });
            auto command = commandBatch(
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement, 7u, entries);
            command.header.accepted_cycles = admitted;
            ASSERT_TRUE(command.valid());
            const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
            ASSERT_EQ(physical.migration_cycles.size(), admitted);
            EXPECT_NO_THROW(requireSimpleMoEOverlayPhysicalCycles(
                physical.migrations, physical.migration_cycles));
            // This calls the genuine alternate host planner; it demonstrates
            // why re-planning authenticated device groups changes lane demand.
            EXPECT_EQ(decomposeMoEOverlayMigrationCycles(physical.migrations).size(), admitted + 1u);
            for (const auto malformed : {0u, admitted + 1u, 1u})
            {
                auto bad = command;
                bad.entries.front().flags = moeOverlayMovementFlags(
                    MoEOverlayDeviceMovementAxis::TierResidency, malformed);
                bad.header.command_digest = commandDigest(bad.entries);
                if (malformed == 1u &&
                    moeOverlayMovementCycle(command.entries.front().flags) == 1u)
                    bad.entries.back().flags = moeOverlayMovementFlags(
                        MoEOverlayDeviceMovementAxis::TierResidency, 1u);
                bad.header.command_digest = commandDigest(bad.entries);
                EXPECT_THROW(makeMoEOverlayDevicePhysicalMovementBatch(bad, topology()),
                             std::invalid_argument);
            }
        }
    }

    /** @test Immutable device commands preserve distinct simple cycles sharing a target. */
    TEST(Test__MoEOverlayDevicePhysicalMovement,
         DurableCommandsRetainSimpleCyclesAcrossSharedTargets)
    {
        for (const auto kind : {
                 MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
                 MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore})
        {
            for (const int first_layer : {0, 31})
            {
                auto command = commandBatch(kind, 7u, {
                    move(MoEOverlayDeviceMovementOp::DurableMove, 1, 0, 2, 0),
                    move(MoEOverlayDeviceMovementOp::DurableMove, 1, 1, 3, 0),
                    move(MoEOverlayDeviceMovementOp::DurableMove, 1, 2, 0, 2),
                    move(MoEOverlayDeviceMovementOp::DurableMove, 1, 3, 0, 3),
                });
                command.first_model_layer = first_layer;
                command.entries[1].flags = moeOverlayMovementFlags(
                    moeOverlayMovementAxis(command.entries[1].flags), 2u);
                command.entries[3].flags = moeOverlayMovementFlags(
                    moeOverlayMovementAxis(command.entries[3].flags), 2u);
                command.header.command_digest = commandDigest(command.entries);
                command.header.accepted_cycles = 2u;
                ASSERT_TRUE(command.valid());
                const auto physical = makeMoEOverlayDevicePhysicalMovementBatch(command, topology());
                ASSERT_TRUE(physical.valid());

                // Both swaps use target 0, but each original admitted cycle
                // visits it only once. Joining them into an Euler circuit
                // invents a repeated-target cycle outside the lane-plan bound.
                EXPECT_NO_THROW(requireSimpleMoEOverlayPhysicalCycles(
                    physical.migrations, physical.migration_cycles));
                ASSERT_EQ(physical.migration_cycles.size(), 2u);
                EXPECT_EQ(physical.migration_cycles[0].migration_indices,
                          (std::vector<std::size_t>{0u, 2u}));
                EXPECT_EQ(physical.migration_cycles[1].migration_indices,
                          (std::vector<std::size_t>{1u, 3u}));
                for (const auto &cycle : physical.migration_cycles)
                    EXPECT_EQ(cycle.layer_idx, first_layer + 1);

                EXPECT_EQ(physical.command_digest, command.header.command_digest);
                EXPECT_EQ(physical.packed_weight_bytes, 4u * 4096u);
                ASSERT_EQ(physical.migrations.size(), command.entries.size());
                for (std::size_t index = 0; index < physical.migrations.size(); ++index)
                {
                    const auto &edge = physical.migrations[index];
                    EXPECT_EQ(edge.expert_id, command.entries[index].expert);
                    EXPECT_EQ(edge.source.owner_participant, command.entries[index].source_participant);
                    EXPECT_EQ(edge.destination.owner_participant, command.entries[index].destination_participant);
                    EXPECT_EQ(edge.axis, MoEOptimizationMovementAxis::TierResidency);
                }
            }
        }
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
