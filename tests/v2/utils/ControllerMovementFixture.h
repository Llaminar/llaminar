/**
 * @file ControllerMovementFixture.h
 * @brief Device-free completed cycles for controller transport regressions.
 *
 * Metadata covers homogeneous and mixed GPU vendors, rank crossings and
 * unequal promotion/demotion counts in a closed three-participant cycle.
 * Actual copied bytes intentionally differ from estimated expert sizes.
 */
#pragma once
#include "execution/moe/MoEOptimizationStatus.h"

namespace llaminar2::test
{
    /**
     * @param transaction Exact completed protocol identity, including values above 2^53.
     * @param epoch Published durable epoch; independent of request-local ordinals.
     * @param geometry 0=CUDA, 1=ROCm, 2=mixed-vendor cycle, 3=two policy cycles merged physically.
     * @return One policy leader's completed physical wave and admitting proof.
     */
    inline MoEOptimizationMovementLedger controllerMovementFixture(
        std::uint64_t transaction, std::uint64_t epoch, unsigned geometry)
    {
        MoEOptimizationMovementLedger ledger;
        const int count = geometry == 2 ? 3 : geometry == 3 ? 4 : 2;
        MoEOptimizationControllerMovementReceipt receipt{
            .base_epoch = epoch - 1, .snapshot_observations = 64, .accepted_cycles = geometry == 3 ? 2u : 1u, .physical_cycles = 1,
            .economy = {1000, 100, 50, 850}, .changed_layers = 1,
            .edges_checked = static_cast<std::uint64_t>(count),
            .participant_coordinates_checked = static_cast<std::uint64_t>(geometry == 3 ? 2 : count),
            .tier_coordinates_checked = static_cast<std::uint64_t>(geometry == 2 ? 3 : 1)};
        for (int i = 0; i < count; ++i)
        {
            const int source_id = geometry == 3 ? i % 2 : i;
            const int next = geometry == 3 ? 1 - source_id : (i + 1) % count;
            const int source_priority = geometry == 2 ? i : 0;
            const int destination_priority = geometry == 2 ? next : 0;
            const auto direction = destination_priority == source_priority ? MoEOptimizationMovementDirection::SamePriority :
                destination_priority < source_priority ? MoEOptimizationMovementDirection::Promotion : MoEOptimizationMovementDirection::Demotion;
            const auto source = geometry == 1 || (geometry == 2 && i == 1) ? DeviceId::rocm(source_id) : DeviceId::cuda(source_id);
            const auto destination = geometry == 1 || (geometry == 2 && next == 1) ? DeviceId::rocm(next) : DeviceId::cuda(next);
            ledger.edges.push_back({.authority = MoEOptimizationAuthority::Device,
                .transaction = transaction, .candidate_epoch = epoch, .layer = 1, .expert = i,
                .cycle_index = 0, .cycle_size = static_cast<std::size_t>(count), .direction = direction,
                .axis = geometry == 2 ? MoEOptimizationMovementAxis::Combined : MoEOptimizationMovementAxis::ParticipantPlacement,
                .source_participant = source_id, .destination_participant = next,
                .source_priority = source_priority, .destination_priority = destination_priority,
                .source_device = source, .destination_device = destination,
                .source_world_rank = source_id, .destination_world_rank = next,
                .source_world_rank_known = true, .destination_world_rank_known = true,
                .estimated_weight_bytes = 8192, .activation_count = static_cast<std::uint64_t>(10 + i)});
            receipt.promotions += direction == MoEOptimizationMovementDirection::Promotion;
            receipt.demotions += direction == MoEOptimizationMovementDirection::Demotion;
            receipt.same_priority_moves += direction == MoEOptimizationMovementDirection::SamePriority;
            receipt.cross_domain_moves += source_priority != destination_priority;
            receipt.cross_rank_moves += 1;
            receipt.cross_backend_moves += source.type != destination.type;
        }
        ledger.device_publications.push_back({transaction, epoch, static_cast<std::uint64_t>(count),
            10007 + transaction % 20, receipt});
        ledger.economy.push_back({.authority = MoEOptimizationAuthority::Device,
            .transaction = transaction, .candidate_epoch = epoch, .command_count = static_cast<std::uint64_t>(count),
            .cycle_count = 1, .proof = receipt.economy});
        return ledger;
    }
}
