/**
 * @file NativeMoEMovementPerfStats.h
 * @brief Bounded mirrors of completed native ownership-swap publications.
 *
 * Immutable archive receipts supply actual copied bytes and exact edge identity.
 * Transaction IDs, epochs and expert IDs are numeric observations, never map
 * keys. All wave counters share one ordered witness, while the edge witness
 * preserves every canonical field for comparison with the terminal archive.
 * Frozen stage layer bounds separate independent controllers, even when they
 * share a physical participant. Only metadata is observed; this helper performs no device or payload I/O.
 */
#pragma once

#include "MoEOptimizationStatus.h"
#include "utils/PerfStatsCollector.h"
#include <span>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Mirror one already-retired native movement wave with constant key count.
     * @param publication Actual physical receipt retained by the native archive.
     * @param edges Exact live edge range belonging to that same completed wave.
     * @param device Stable authority participant label.
     * @param first_model_layer First global layer owned by this captured controller.
     * @param layer_count Compact layer capacity frozen with its runtime table.
     * @throws std::invalid_argument before emitting any counter for an invalid wave.
     *
     * Wire words are explicitly numbered independently of C++ enum ordinals:
     * backend 1=CUDA, 2=ROCm; direction 0=same priority; axis 1=participant
     * placement. Signed priorities use their uint64 two's-complement value.
     * Unknown ranks encode a false presence word followed by zero. This is the
     * semantic JSON identity, not struct padding or a memory image.
     */
    inline void recordNativeMoECompletedMovement(
        const MoEOptimizationDeviceMovementPublication &publication,
        std::span<const MoEOptimizationMovementEdge> edges, const std::string &device,
        int first_model_layer, std::uint32_t layer_count)
    {
        if (first_model_layer < 0 || layer_count == 0 || layer_count > INT32_MAX - first_model_layer ||
            !publication.valid() || publication.controller || publication.command_count != edges.size() || edges.size() % 2 != 0)
            throw std::invalid_argument("Native movement telemetry requires one complete physical wave");
        for (const auto &edge : edges)
            if (!edge.valid() || edge.layer < first_model_layer ||
                static_cast<std::uint32_t>(edge.layer - first_model_layer) >= layer_count ||
                edge.authority != MoEOptimizationAuthority::Device ||
                edge.transaction != publication.transaction || edge.candidate_epoch != publication.candidate_epoch ||
                edge.direction != MoEOptimizationMovementDirection::SamePriority ||
                edge.axis != MoEOptimizationMovementAxis::ParticipantPlacement ||
                edge.source_priority != edge.destination_priority || edge.blocking_inference ||
                edge.estimated_weight_bytes == 0 || !edge.source_device.is_gpu() || !edge.destination_device.is_gpu() ||
                edge.source_device.type != edge.destination_device.type ||
                (edge.source_world_rank_known && edge.source_world_rank < 0) ||
                (edge.destination_world_rank_known && edge.destination_world_rank < 0))
                throw std::invalid_argument("Native movement telemetry edge disagrees with its completed wave");
        if (!PerfStatsCollector::isDomainEnabled("moe_overlay_controller"))
            return;

        const PerfStatsCollector::Tags tags{{"policy_owner", "device"},
            {"policy", "native_load_spread"}, {"encoding", "completed_wave_v1"},
            {"first_model_layer", std::to_string(first_model_layer)}, {"layer_count", std::to_string(layer_count)}};
        const auto words = {publication.transaction, publication.candidate_epoch,
            publication.command_count, publication.physical_payload_bytes};
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller", "dynamic_movement_transactions",
            1.0, words, "maintenance", device, tags);
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller", "dynamic_physical_bytes",
            static_cast<double>(publication.physical_payload_bytes), words, "maintenance", device, tags);
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller", "dynamic_migration_edges",
            static_cast<double>(publication.command_count), words, "maintenance", device, tags);
        for (const auto &edge : edges)
            PerfStatsCollector::addCounterWithSequence("moe_overlay_controller", "dynamic_migration_edge_identities", 1.0,
                {edge.transaction, edge.candidate_epoch,
                 static_cast<std::uint64_t>(edge.layer), static_cast<std::uint64_t>(edge.expert),
                 edge.cycle_index, edge.cycle_size, 0, 1,
                 static_cast<std::uint64_t>(edge.source_participant), static_cast<std::uint64_t>(edge.destination_participant),
                 static_cast<std::uint64_t>(static_cast<std::int64_t>(edge.source_priority)),
                 static_cast<std::uint64_t>(static_cast<std::int64_t>(edge.destination_priority)),
                 edge.source_device.is_cuda() ? 1u : 2u, static_cast<std::uint64_t>(edge.source_device.ordinal),
                 edge.destination_device.is_cuda() ? 1u : 2u, static_cast<std::uint64_t>(edge.destination_device.ordinal),
                 edge.source_world_rank_known ? 1u : 0u,
                 edge.source_world_rank_known ? static_cast<std::uint64_t>(edge.source_world_rank) : 0u,
                 edge.destination_world_rank_known ? 1u : 0u,
                 edge.destination_world_rank_known ? static_cast<std::uint64_t>(edge.destination_world_rank) : 0u,
                 edge.estimated_weight_bytes, edge.activation_count, 0}, "maintenance", device, tags);
    }
}
