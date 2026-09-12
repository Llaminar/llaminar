/**
 * @file MoEMovementLedgerJson.h
 * @brief Lossless HTTP projection of the ExpertOverlay owner's completed ledger.
 *
 * The caller obtains one immutable snapshot from IOrchestrationRunner after
 * generation, before request cleanup. This serializer cannot access a runner,
 * controller, device, or PerfStats. It neither advances maintenance nor turns
 * proposals into completed movement. All records retain their model-lifetime
 * scope: consecutive responses overlap and must never be summed as disjoint
 * request counters. Ordinary responses and logging do not invoke this export.
 */
#pragma once

#include "execution/moe/MoEOptimizationStatus.h"
#include "execution/moe/MoEOptimizationMovementTopology.h"
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace llaminar2::movement_json_detail
{
    /**
     * @param authority Actual placement-policy owner from the immutable record.
     * @return Stable wire identity; unset/invalid owners throw std::invalid_argument.
     */
    inline const char *authorityName(MoEOptimizationAuthority authority)
    {
        switch (authority)
        {
        case MoEOptimizationAuthority::Host: return "host";
        case MoEOptimizationAuthority::Device: return "device";
        case MoEOptimizationAuthority::None: break;
        }
        throw std::invalid_argument("Completed movement has no valid authority");
    }

    /**
     * @param direction Owner-authored physical movement classification.
     * @return Wire direction independent of objective; invalid enums throw std::invalid_argument.
     */
    inline const char *directionName(MoEOptimizationMovementDirection direction)
    {
        switch (direction)
        {
        case MoEOptimizationMovementDirection::Promotion: return "promotion";
        case MoEOptimizationMovementDirection::Demotion: return "demotion";
        case MoEOptimizationMovementDirection::SamePriority: return "same_priority";
        }
        throw std::invalid_argument("Completed movement has an invalid direction");
    }

    /**
     * @param axis Owner-authored logical objective, not inferred from priorities.
     * @return Wire objective; invalid enums throw std::invalid_argument.
     */
    inline const char *axisName(MoEOptimizationMovementAxis axis)
    {
        switch (axis)
        {
        case MoEOptimizationMovementAxis::TierResidency: return "tier_residency";
        case MoEOptimizationMovementAxis::ParticipantPlacement: return "participant_placement";
        case MoEOptimizationMovementAxis::Combined: return "combined";
        }
        throw std::invalid_argument("Completed movement has an invalid axis");
    }

    /**
     * @param counts Existing exclusive cycle buckets in the policy proof.
     * @return Exact classification without counting a combined objective twice.
     */
    inline nlohmann::json axisCounts(const MoEOptimizationCycleAxisCounts &counts)
    {
        return {{"tier_residency", counts.tier_residency},
                {"participant_placement", counts.participant_placement},
                {"combined", counts.combined}};
    }

    /**
     * @brief Project an existing durable edge without inventing transfer facts.
     * @param edge Immutable placement publication from the sole authority.
     * @return Complete edge identity; unknown MPI ranks remain explicit nulls.
     * @throws std::invalid_argument if the owner supplied an invalid record.
     */
    inline nlohmann::json edgeRecord(const MoEOptimizationMovementEdge &edge)
    {
        if (!edge.valid())
            throw std::invalid_argument("Completed movement ledger contains an invalid edge");
        // Preserve the original estimated-byte label. A published edge proves
        // completion, but its estimate must not masquerade as a wire-byte meter.
        return {{"authority", authorityName(edge.authority)},
                {"transaction", edge.transaction}, {"candidate_epoch", edge.candidate_epoch},
                {"layer", edge.layer}, {"expert", edge.expert},
                {"cycle_index", edge.cycle_index}, {"cycle_size", edge.cycle_size},
                {"direction", directionName(edge.direction)}, {"movement_axis", axisName(edge.axis)},
                {"source_participant", edge.source_participant},
                {"destination_participant", edge.destination_participant},
                {"source_priority", edge.source_priority},
                {"destination_priority", edge.destination_priority},
                {"source_device", edge.source_device.toString()},
                {"destination_device", edge.destination_device.toString()},
                {"source_world_rank", edge.source_world_rank_known
                    ? nlohmann::json(edge.source_world_rank) : nlohmann::json(nullptr)},
                {"destination_world_rank", edge.destination_world_rank_known
                    ? nlohmann::json(edge.destination_world_rank) : nlohmann::json(nullptr)},
                {"estimated_weight_bytes", edge.estimated_weight_bytes},
                {"activation_count", edge.activation_count},
                {"blocking_inference", edge.blocking_inference}};
    }

    /**
     * @brief Project the admitting owner's exact completed-wave economy proof.
     * @param record Policy evidence published atomically with the move edges.
     * @return Explicit policy units and the original integer decision inputs.
     * @throws std::invalid_argument if the authority proof is inconsistent.
     */
    inline nlohmann::json economyRecord(const MoEOptimizationMovementEconomy &record)
    {
        if (!record.valid())
            throw std::invalid_argument("Completed movement ledger contains invalid economy evidence");
        nlohmann::json result = {{"authority", authorityName(record.authority)},
                {"transaction", record.transaction}, {"candidate_epoch", record.candidate_epoch},
                {"command_count", record.command_count}, {"cycle_count", record.cycle_count}};
        if (const auto *time = std::get_if<MoEOptimizationTimeEconomy>(&record.proof))
        {
            result.update({{"policy", "time_ns"},
                {"projected_service_gain_ns", time->projected_service_gain_ns},
                {"projected_transfer_and_repack_ns", time->projected_transfer_and_repack_ns},
                {"projected_inference_interference_ns", time->projected_inference_interference_ns},
                {"projected_net_benefit_ns", time->projected_net_benefit_ns}});
        }
        else
        {
            const auto &load = std::get<DeviceMoERebalanceLoadSpreadProof>(record.proof);
            result.update({{"policy", "native_load_spread"},
                {"accepted_spread_improvement", load.accepted_spread_improvement},
                {"pre_wave_spread", load.pre_wave_spread}, {"post_wave_spread", load.post_wave_spread},
                {"pre_wave_total", load.pre_wave_total}, {"post_wave_total", load.post_wave_total},
                {"pre_participant_spread", load.pre_participant_spread},
                {"post_participant_spread", load.post_participant_spread},
                {"pre_participant_total", load.pre_participant_total},
                {"post_participant_total", load.post_participant_total},
                {"requested_payload_slots", load.requested_payload_slots},
                {"minimum_improvement_per_slot", load.minimum_improvement_per_slot},
                {"maximum_post_spread_per_mille", load.maximum_post_spread_per_mille},
                {"ownership_swap_accepts", load.ownership_swap_accepts}});
        }
        return result;
    }

    /**
     * @brief Preserve the host owner's capacity and two-axis admission proof.
     * @param record Completed host-authoritative wave classification.
     * @return Every original admission field; device authorities need no host record.
     * @throws std::invalid_argument if the authority proof is inconsistent.
     */
    inline nlohmann::json admissionRecord(const MoEOptimizationHostMovementAdmission &record)
    {
        if (!record.valid())
            throw std::invalid_argument("Completed movement ledger contains invalid host admission evidence");
        return {{"authority", authorityName(record.authority)},
                {"transaction", record.transaction}, {"candidate_epoch", record.candidate_epoch},
                {"cycle_capacity_kind", record.cycle_capacity_kind ==
                    MoEOptimizationCycleCapacityKind::Bounded ? "bounded" : "unbounded"},
                {"maximum_concurrent_cycles", record.maximum_concurrent_cycles},
                {"candidate_cycles", record.candidate_cycles},
                {"policy_eligible_cycles", record.policy_eligible_cycles},
                {"policy_eligible_axes", axisCounts(record.policy_eligible_axes)},
                {"admitted_candidate_cycles", record.admitted_candidate_cycles},
                {"admitted_candidate_axes", axisCounts(record.admitted_candidate_axes)},
                {"admitted_physical_cycles", record.admitted_physical_cycles},
                {"admitted_physical_axes", axisCounts(record.admitted_physical_axes)},
                {"individual_policy_rejected_cycles", record.individual_policy_rejected_cycles},
                {"dependent_payoff_rejected_cycles", record.dependent_payoff_rejected_cycles},
                {"capacity_rejected_cycles", record.capacity_rejected_cycles},
                {"participant_axis_budget_rejected_cycles", record.participant_axis_budget_rejected_cycles},
                {"dependent_cohort_candidates", record.dependent_cohort_candidates},
                {"dependent_cohort_payoff_rejections", record.dependent_cohort_payoff_rejections},
                {"physical_cycle_recomposition", record.physical_cycle_recomposition},
                {"capacity_bounded", record.capacity_bounded},
                {"policy_bounded", record.policy_bounded}};
    }
}

namespace llaminar2
{
    /**
     * @brief Project immutable admitted geometry separately from completed moves.
     * @param topology Canonical frozen-plan description; never observed-axis inference.
     * @return Required owner and available logical axes for the model lifetime.
     * @throws std::invalid_argument for inconsistent or unknown geometry values.
     */
    inline nlohmann::json moeMovementTopologyJson(const MoEOptimizationMovementTopology &topology)
    {
        if (!topology.valid())
            throw std::invalid_argument("Movement topology contains an invalid authority or axes");
        auto axes = nlohmann::json::array();
        if (hasTierResidencyAxis(topology.axes))
            axes.push_back("tier_residency");
        if (hasParticipantPlacementAxis(topology.axes))
            axes.push_back("participant_placement");
        return {{"schema", 1}, {"scope", "model_lifetime"},
            {"authority", topology.authority == MoEOptimizationAuthority::None
                ? "none" : movement_json_detail::authorityName(topology.authority)},
            {"available_axes", std::move(axes)}};
    }

    /**
     * @brief Serialize one complete passive model-lifetime movement snapshot.
     * @param ledger Existing authority snapshot, acquired exactly once by the caller.
     * @return Versioned evidence for an opt-in terminal HTTP response.
     * @throws std::invalid_argument on truncated or internally invalid evidence.
     *
     * The three arrays preserve owner publication order and retain identities
     * needed to distinguish objective, physical direction and capacity cycles.
     * No deduplication, counter summation or inferred policy occurs here.
     */
    inline nlohmann::json moeMovementLedgerJson(const MoEOptimizationMovementLedger &ledger)
    {
        if (!ledger.complete())
            throw std::invalid_argument("Completed movement ledger is truncated or has conflicting authorities");
        nlohmann::json result = {{"schema", 2}, {"scope", "model_lifetime"},
            {"complete", true}, {"discarded_edges", ledger.discarded_edges},
            {"discarded_economy_records", ledger.discarded_economy_records},
            {"discarded_host_admission_records", ledger.discarded_host_admission_records},
            {"edges", nlohmann::json::array()}, {"economy", nlohmann::json::array()},
            {"host_admissions", nlohmann::json::array()}};
        // Pre-size array capacity: opt-in diagnostics may span many epochs,
        // but must not repeatedly move a growing journal while serializing it.
        auto &edges = result["edges"].get_ref<nlohmann::json::array_t &>();
        auto &economy = result["economy"].get_ref<nlohmann::json::array_t &>();
        auto &admissions = result["host_admissions"].get_ref<nlohmann::json::array_t &>();
        edges.reserve(ledger.edges.size());
        economy.reserve(ledger.economy.size());
        admissions.reserve(ledger.host_admissions.size());
        for (const auto &edge : ledger.edges)
            edges.push_back(movement_json_detail::edgeRecord(edge));
        for (const auto &record : ledger.economy)
            economy.push_back(movement_json_detail::economyRecord(record));
        for (const auto &record : ledger.host_admissions)
            admissions.push_back(movement_json_detail::admissionRecord(record));
        return result;
    }
}
