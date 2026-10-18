/**
 * @file MoEOverlayControllerPerfStats.h
 * @brief Bounded observations of completed device-controller maintenance.
 *
 * The transport follower already owns an authenticated immutable command.
 * These publishers retain its numeric lifecycle evidence without adding a
 * transaction key, changing device policy or reading mutable GPU state.
 */
#pragma once

#include "execution/moe/MoEOverlayDeviceControllerABI.h"
#include "utils/PerfStatsCollector.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /**
     * @brief Record an admitted maintenance window without indexing by its size.
     * @param completed_tokens Observations coalesced into this notification.
     * @param required_tokens Threshold admitted by the existing cadence authority.
     * @param phase Stable prefill/decode source of the notification.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEMaintenanceNotification(
        std::uint64_t completed_tokens, std::uint64_t required_tokens,
        const char *phase, const std::string &device)
    {
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "background_notification_batches", 1.0, {completed_tokens, required_tokens},
            "maintenance", device, {{"phase", phase}, {"blocking_inference", "false"}});
    }

    /**
     * @brief Preserve an adaptive cadence transition within one fixed policy key.
     * @param previous_tokens Completed window's admitted threshold.
     * @param next_tokens Threshold selected for the following window.
     * @param maximum_tokens Immutable configured upper bound.
     * @param growth_factor Immutable configured growth policy.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEMaintenanceWindowGrowth(
        std::uint64_t previous_tokens, std::uint64_t next_tokens,
        std::uint64_t maximum_tokens, double growth_factor, const std::string &device)
    {
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "maintenance_window_growth", 1.0, {previous_tokens, next_tokens},
            "maintenance", device, {{"maximum_tokens", std::to_string(maximum_tokens)},
                {"growth_factor", std::to_string(growth_factor)},
                {"policy_owner", "device"}, {"scheduler_role", "retained_graph_submission"}});
    }

    /**
     * @brief Record service samples from one participant and inference phase.
     * @param participant Frozen topology participant ID.
     * @param phase Stable prefill/decode calibration source.
     * @param samples Device-authored sample count already observed by the follower.
     * @param activations Routed activations represented by those samples.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEServicePhaseObservation(
        int participant, const char *phase, std::uint64_t samples,
        std::uint64_t activations, const std::string &device)
    {
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "device_service_phase_observed", 1.0, {samples, activations}, "maintenance", device,
            {{"participant_id", std::to_string(participant)}, {"source", phase},
             {"blocking_inference", "false"}, {"evidence_source", "device_local"}});
    }

    /**
     * @brief Record the exact service-snapshot publication generation.
     * @param participant Frozen topology participant ID.
     * @param generation Authenticated immutable snapshot generation.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEServiceSnapshotPublication(
        int participant, std::uint64_t generation, const std::string &device)
    {
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "device_service_snapshot_publications", 1.0, {generation}, "maintenance", device,
            {{"participant_id", std::to_string(participant)},
             {"blocking_inference", "false"}, {"evidence_source", "device_local"}});
    }

    /**
     * @brief Count imported service rows while retaining their exact sample extent.
     * @param rows Number of imported immutable metadata rows.
     * @param samples Device observations represented by those rows.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEServiceSnapshotRows(
        std::uint64_t rows, std::uint64_t samples, const std::string &device)
    {
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "device_service_snapshot_rows", static_cast<double>(rows), {rows, samples},
            "maintenance", device, {{"blocking_inference", "false"}, {"evidence_source", "device_local"}});
    }

    /** @brief Immutable physical-progress counters from one completed transport wave. */
    struct MoEPhysicalWaveProgressEvidence
    {
        std::uint64_t transaction; ///< Authenticated movement command identity.
        std::uint64_t migrations; ///< Exact migrations represented by this wave.
        std::uint64_t operations; ///< Independent projection operations submitted.
        std::uint64_t polls; ///< Actual operation readiness polls performed.
        std::uint64_t quanta; ///< Completed bounded host polling quanta.
        std::uint64_t maximum_polls_per_quantum; ///< Installed progress budget.
    };

    /**
     * @brief Preserve physical parallelism and polling cost without per-wave keys.
     * @param evidence Actual counters after every physical projection is ready.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEPhysicalWaveProgress(
        const MoEPhysicalWaveProgressEvidence &evidence, const std::string &device)
    {
        const auto words = {evidence.transaction, evidence.migrations, evidence.operations,
                            evidence.polls, evidence.quanta, evidence.maximum_polls_per_quantum};
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "physical_wave_parallel_operations_started", static_cast<double>(evidence.operations),
            words, "maintenance", device, {{"serialized_submission", "false"}});
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "physical_wave_bounded_progress_polls", static_cast<double>(evidence.polls),
            words, "maintenance", device, {{"driver_fair", "true"}});
    }

    /**
     * @brief Retain completed restoration work separately from optimization work.
     * @param command Authenticated restoration command after physical retirement.
     * @param device Stable owning participant/topology label.
     * @throws std::invalid_argument for a non-restoration or zero-work command.
     */
    inline void recordMoEPreparedContextMovement(
        const MoEOverlayDeviceControllerCommandHeader &command, const std::string &device)
    {
        if (command.kind != static_cast<std::uint32_t>(MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore)
            || command.transaction_id == 0 || command.base_epoch == 0
            || command.candidate_epoch <= command.base_epoch
            || command.candidate_epoch - command.base_epoch != 1
            || command.command_count == 0 || command.packed_weight_bytes == 0)
            throw std::invalid_argument("Restoration movement evidence requires completed physical restoration");
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "prepared_context_restore_movement_waves", 1.0,
            {command.transaction_id, command.base_epoch, command.candidate_epoch,
             command.command_count, command.packed_weight_bytes}, "model_teardown", device,
            {{"policy_owner", "device"}, {"excluded_from_optimization_ledger", "true"}});
    }

    /**
     * @brief Record a completed no-movement decision or exact restoration check.
     * @param command Immutable command after every protocol participant completed.
     * @param device Stable owning participant/topology label.
     * @throws std::invalid_argument if this is not a zero-work durable command.
     *
     * Restoration stays a distinct teardown family, so disposal cannot masquerade
     * as useful optimization. Every accepted decision uses one constant-width
     * sequence step even when its histogram, scan cursor and cost values change.
     */
    inline void recordMoEZeroMovementCompletion(
        const MoEOverlayDeviceControllerCommandHeader &command, const std::string &device)
    {
        const auto kind = static_cast<MoEOverlayDeviceControllerTransactionKind>(command.kind);
        if (command.transaction_id == 0 || command.base_epoch == 0
            || command.candidate_epoch != command.base_epoch
            || command.command_count != 0 || command.packed_weight_bytes != 0
            || (kind != MoEOverlayDeviceControllerTransactionKind::DynamicPlacement
                && kind != MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore))
            throw std::invalid_argument("No-movement evidence requires a completed zero-work durable command");
        if (kind == MoEOverlayDeviceControllerTransactionKind::PreparedContextRestore)
        {
            PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
                "prepared_context_restore_certifications", 1.0,
                {command.transaction_id, command.candidate_epoch}, "model_teardown", device,
                {{"movement_commands", "0"}, {"policy_owner", "device"},
                 {"exact_initial_owner_table", "true"}});
            return;
        }
        PerfStatsCollector::addCounterWithSequence("moe_overlay_controller",
            "dynamic_zero_movement_transactions", 1.0,
            {command.transaction_id, command.base_epoch, command.candidate_epoch,
             command.snapshot_observations, command.priority_cost_before, command.priority_cost_after,
             command.same_priority_makespan_before, command.same_priority_makespan_after,
             command.accepted_cycles, command.rejected_cycles, command.phase_tradeoff_candidates,
             command.improvement_floor_rejected_cycles, command.payoff_rejected_cycles,
             command.residency_rejected_cycles, command.projected_service_gain_ns,
             command.projected_transfer_and_repack_ns, command.projected_inference_interference_ns,
             command.projected_net_benefit_ns, command.layer_scan_start, command.layer_scan_next},
            "maintenance", device, {{"movement_commands", "0"}, {"physical_bytes", "0"},
                {"bounded_device_phases", "true"}, {"resident_external_waits", "0"},
                {"prearmed_cross_device_fanin", "true"}, {"policy_owner", "device"}});
    }
}
