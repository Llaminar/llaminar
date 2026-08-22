/**
 * @file MoEOverlayDevicePlacementPolicy.h
 * @brief CPU oracle for topology-wide device-owned ExpertOverlay placement.
 *
 * Production CUDA and HIP policy kernels consume mapped device snapshots and
 * must emit the same canonical command sequence as this pointer-free reference.
 * The reference is used by fast unit tests and real-device differential tests;
 * it is never an inference-time host authority.  Policy preserves every
 * participant's per-layer expert quota, prefers lower integer priorities, and
 * then reduces participant makespan within equal-priority tiers.
 */

#pragma once

#include "DeviceMoERebalancePolicyShared.h"
#include "MoEOverlayDeviceControllerFabricABI.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace llaminar2
{
    /** Immutable logical identity needed to score one overlay participant. */
    struct MoEOverlayDevicePlacementParticipant
    {
        std::uint32_t participant_id = 0u; ///< Dense controller participant id.
        std::int32_t tier_priority = 0;    ///< Lower integer means preferred tier.
        std::int32_t tier_index = -1;      ///< Dense measured-service tier.

        /** @return Whether this record occupies its canonical dense index. */
        [[nodiscard]] constexpr bool validAt(std::size_t index) const noexcept
        {
            return participant_id == index;
        }
    };

    /** Complete immutable measured costs and histories for the CPU oracle. */
    struct MoEOverlayDevicePlacementEconomyInput
    {
        std::uint32_t tier_count = 0u;
        std::uint32_t routed_experts_per_token = 0u;
        /** Device transaction/generation used by committed hysteresis. */
        std::uint64_t transaction_generation = 0u;
        /** `[decode,prefill][layer][expert]` cumulative routed demand. */
        std::vector<std::uint64_t> phase_expert_demand;
        /** `[tier][layer][decode,prefill,grouped]` ns per activation. */
        std::vector<std::uint64_t> service_costs;
        /** Dense `[source][destination][layer]` directed movement prices. */
        std::vector<MoEOverlayDeviceControllerMigrationCost> migration_costs;
        /** `[layer][expert]` committed movement generation or never sentinel. */
        std::vector<std::uint64_t> last_moved_generation;
        std::uint64_t minimum_residency_generations = 0u;
        std::uint64_t payoff_horizon_tokens = 0u;
        std::uint64_t minimum_net_benefit_ns = 0u;

        /** @return Whether every measured coordinate matches policy geometry. */
        [[nodiscard]] bool valid(
            std::uint32_t num_layers,
            std::uint32_t num_experts,
            std::uint32_t participant_count) const noexcept;
    };

    /** Complete immutable input to one durable Dynamic policy wave. */
    struct MoEOverlayDevicePlacementPolicyInput
    {
        std::uint32_t num_layers = 0u;
        std::uint32_t num_experts = 0u;
        std::vector<MoEOverlayDevicePlacementParticipant> participants;

        /**
         * Participant-major, layer-major, expert-major snapshot words.
         * Every word is produced by
         * `moe_rebalance_policy::packCollectedState`; exactly one participant
         * must carry the authoritative-owner bit for each layer/expert.
         */
        std::vector<std::uint64_t> collected_state;

        /** Exact complete packed GPU expert bytes for each model layer. */
        std::vector<std::uint64_t> payload_bytes_per_layer;

        std::uint64_t base_epoch = 0u; ///< Durable epoch sampled by the snapshot.
        /** Device-owned first layer for this bounded cyclic policy scan. */
        std::uint32_t layer_start_cursor = 0u;
        std::uint64_t minimum_window_activations = 1u;
        std::uint32_t maximum_cycles_per_wave = 1u;
        std::uint32_t dynamic_imbalance_threshold_per_mille =
            moe_rebalance_policy::
                kDefaultDynamicImbalanceThresholdPerMille;
        std::uint32_t dynamic_minimum_improvement_per_mille =
            moe_rebalance_policy::
                kDefaultDynamicMinImprovementPerMille;
        std::uint32_t dynamic_maximum_cycles_per_layer =
            moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer;
        /** Zero means @ref command_capacity, matching the device ABI. */
        std::uint32_t dynamic_maximum_commands_per_wave =
            moe_rebalance_policy::kDefaultDynamicMaxPlanEntriesPerWave;
        std::uint32_t command_capacity = 0u;
        /** Required for measured-economy differential certification. */
        std::optional<MoEOverlayDevicePlacementEconomyInput> economy;

        /** @return Whether geometry, storage extents, and epoch are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** Auditable before/after objective and movement classification. */
    struct MoEOverlayDevicePlacementPolicyEvidence
    {
        std::uint64_t snapshot_observations = 0u;
        std::uint64_t priority_cost_before = 0u;
        std::uint64_t priority_cost_after = 0u;
        std::uint64_t same_priority_makespan_before = 0u;
        std::uint64_t same_priority_makespan_after = 0u;
        std::uint32_t accepted_cycles = 0u;
        std::uint32_t rejected_cycles = 0u;
        std::uint32_t promotions = 0u;
        std::uint32_t demotions = 0u;
        std::uint32_t same_priority_moves = 0u;
        std::uint32_t changed_layers = 0u;
        /** First layer examined by this policy wave. */
        std::uint32_t layer_scan_start = 0u;
        /** Cursor that the sole authority must retain for the next wave. */
        std::uint32_t layer_scan_next = 0u;
        std::uint64_t projected_service_gain_ns = 0u;
        std::uint64_t projected_transfer_and_repack_ns = 0u;
        std::uint64_t projected_inference_interference_ns = 0u;
        std::uint64_t projected_net_benefit_ns = 0u;
        std::uint32_t payoff_rejected_cycles = 0u;
        std::uint32_t residency_rejected_cycles = 0u;

        /** @return Whether the accepted plan improves the lexicographic objective. */
        [[nodiscard]] bool improves() const noexcept;
    };

    /** Canonical command batch and resulting owner map for one candidate epoch. */
    struct MoEOverlayDevicePlacementPolicyPlan
    {
        std::vector<MoEOverlayDeviceMovementCommand> commands;
        /** Layer-major/expert-major authoritative owner after accepted cycles. */
        std::vector<std::int32_t> candidate_owner;
        MoEOverlayDevicePlacementPolicyEvidence evidence;

        /** @return Whether a non-empty capacity-preserving candidate was found. */
        [[nodiscard]] bool hasMovement() const noexcept
        {
            return !commands.empty() && evidence.accepted_cycles > 0u;
        }
    };

    /**
     * @brief Deterministic CPU specification for the device Dynamic planner.
     *
     * The planner first assigns the hottest experts to participant quotas in
     * ascending priority order. Within one priority it greedily minimizes
     * current makespan while preserving every participant's exact expert
     * cardinality. The resulting balanced directed move graph is decomposed
     * into complete cycles; only individually improving cycles enter a bounded
     * one-round wave. This makes a partially applied epoch impossible.
     */
    class MoEOverlayDevicePlacementPolicyReference final
    {
    public:
        /**
         * @brief Build one bounded two-axis Dynamic candidate.
         * @param input Authenticated snapshot, topology, epoch, and byte geometry.
         * @return Canonically sorted logical movement commands and evidence.
         * @throws std::invalid_argument for malformed geometry or ownership.
         * @throws std::overflow_error when the candidate epoch cannot advance.
         */
        [[nodiscard]] static MoEOverlayDevicePlacementPolicyPlan planDynamic(
            const MoEOverlayDevicePlacementPolicyInput &input);
    };
} // namespace llaminar2
