/**
 * @file DeviceMoERebalanceLoadSpreadProof.h
 * @brief Immutable, unit-preserving proof of a native GPU placement decision.
 *
 * Native homogeneous maintenance admits transfers using routed-load spread and
 * payload-slot floors, not a measured nanosecond model. The planner seals these
 * inputs with its command generation. Publication and terminal diagnostics can
 * recheck the same pure policy without estimating costs or becoming a second
 * placement authority. This small wire value is shared by CPU, CUDA and HIP.
 */
#pragma once

#include "DeviceMoERebalancePolicyShared.h"
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_PROOF_HD __host__ __device__
#else
#define LLAMINAR_MOE_PROOF_HD
#endif

namespace llaminar2
{
    /**
     * @brief Exact integer inputs that admitted one ownership-swap wave.
     *
     * Wave spread sums the critical-path spreads of serial layers; participant
     * spread instead compares totals across participants. Opposite layer skews
     * can cancel in participant totals, so these are deliberately distinct.
     * All load fields count routed work. No field denotes time or wire bytes.
     */
    struct DeviceMoERebalanceLoadSpreadProof
    {
        std::uint64_t accepted_spread_improvement = 0; ///< Sum of accepted swap gains.
        std::uint64_t pre_wave_spread = 0; ///< Sum of layer spreads before movement.
        std::uint64_t post_wave_spread = 0; ///< Sum of layer spreads after movement.
        std::uint64_t pre_wave_total = 0; ///< Routed work before movement.
        std::uint64_t post_wave_total = 0; ///< Same work redistributed, never removed.
        std::uint64_t pre_participant_spread = 0; ///< Spread of participant totals.
        std::uint64_t post_participant_spread = 0; ///< Proposed participant-total spread.
        std::uint64_t pre_participant_total = 0; ///< Total work in the participant view.
        std::uint64_t post_participant_total = 0; ///< Conserved participant-view work.
        std::uint32_t requested_payload_slots = 0; ///< Physical slot width paid by the wave.
        std::uint32_t minimum_improvement_per_slot = 0; ///< Exact configured load floor.
        std::uint32_t maximum_post_spread_per_mille = 0; ///< Exact configured spread ceiling.
        std::uint32_t ownership_swap_accepts = 0; ///< Accepted closed ownership swaps.

        /**
         * @return True only for a nonempty ownership wave admitted by every
         * original native policy predicate; this does not prove byte transfer.
         */
        [[nodiscard]] LLAMINAR_MOE_PROOF_HD bool valid() const noexcept
        {
            using namespace moe_rebalance_policy;
            // A zero-work policy poll is valid execution, but cannot certify a
            // completed optimization transaction. Rejected commands have none
            // of this positive proof and must not acquire one at publication.
            return requested_payload_slots > 0 && ownership_swap_accepts > 0 &&
                accepted_spread_improvement > 0 &&
                transferWaveMeetsSpreadImprovementFloor(
                    accepted_spread_improvement, requested_payload_slots,
                    minimum_improvement_per_slot) &&
                transferWaveImprovesAggregateLoadSpread(
                    pre_wave_spread, post_wave_spread, pre_wave_total,
                    post_wave_total, requested_payload_slots) &&
                transferWaveParticipantSpreadIsAcceptable(
                    pre_participant_spread, post_participant_spread,
                    pre_participant_total, post_participant_total,
                    requested_payload_slots, post_wave_spread < pre_wave_spread) &&
                transferWaveMeetsPostLoadSpreadCeiling(
                    post_wave_spread, post_wave_total, requested_payload_slots,
                    maximum_post_spread_per_mille);
        }

        /** @brief Compare all immutable inputs, including disabled thresholds. */
        bool operator==(const DeviceMoERebalanceLoadSpreadProof &) const = default;
    };

    static_assert(sizeof(DeviceMoERebalanceLoadSpreadProof) == 88);
    static_assert(std::is_trivially_copyable_v<DeviceMoERebalanceLoadSpreadProof>);

    /** Which admitting policy sealed a reusable native command generation. */
    enum class DeviceMoERebalanceMovementDecisionKind : std::uint32_t
    {
        None, ///< Residency restoration/assignment, not an optimization receipt.
        NativeOwnershipSpread, ///< Paid Dynamic ownership swaps in one native domain.
    };

    /** @brief Wire-stable discriminator and immutable native admission equation. */
    struct DeviceMoERebalanceMovementDecision
    {
        DeviceMoERebalanceMovementDecisionKind kind = DeviceMoERebalanceMovementDecisionKind::None;
        std::uint32_t reserved = 0;
        DeviceMoERebalanceLoadSpreadProof load;
    };

    static_assert(sizeof(DeviceMoERebalanceMovementDecision) == 96);
}

#undef LLAMINAR_MOE_PROOF_HD
