/**
 * @file DeviceMoERebalanceABI.h
 * @brief Minimal cross-backend ABI constants for device-owned MoE rebalance.
 *
 * CUDA and ROCm kernel translation units must validate their private device
 * views against the host controller without including the host tensor, SIMD,
 * and MPI dependency graph. This header is the deliberately tiny common
 * boundary. Keep policy decisions in DeviceMoERebalancePolicyShared.h and keep
 * concrete host records in DeviceMoERebalanceController.h.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace llaminar2::moe_rebalance_abi
{
    /**
     * @brief Version of the fixed host/CUDA/ROCm publication records.
     *
     * Any field-layout change to a cross-backend record must increment this
     * value and update the corresponding byte-size assertion below.
     */
    inline constexpr uint32_t kVersion = 12u;

    /**
     * @brief Exact byte size of one device-owned rebalance command.
     *
     * Ownership-transfer commands carry both the domain-local destination and
     * its overlay-wide route identity. Keeping the latter in the immutable
     * command prevents apply kernels from guessing that two unrelated
     * participant namespaces happen to use the same integer.
     */
    inline constexpr uint32_t kPlanEntryBytes = 52u;

    /**
     * @brief Exact byte size of DeviceMoERebalanceConfig and device views.
     *
     * Active occupancy and directory addressability are deliberately separate
     * fields.  A transfer arrival may promote any staging-origin slot into a
     * durable resident slot, so every backend must preserve both capacities in
     * exactly the same ABI order.
     */
    inline constexpr uint32_t kConfigBytes = 140u;

    /**
     * @brief Exact byte size of DeviceMoERebalanceStatus and device views.
     *
     * The host type and both accelerator views statically assert this value.
     * A diagnostic field can therefore never shift only one backend silently.
     */
    inline constexpr uint32_t kStatusBytes = 648u;
    /** Tail-padding slot used for resident non-owner assignment evidence. */
    inline constexpr uint32_t
        kPrefillCurrentBatchNonOwnerAssignmentLayersOffset = 644u;

    /**
     * @brief Identify payload movement planned from the current routed batch.
     *
     * Prefix-runtime rehydration and steady-state decode maintenance share the
     * same graph-owned transfer machinery. This semantic tag survives command
     * projection so the device apply kernel can publish unambiguous coverage
     * evidence without consulting the host or inferring intent from mutable
     * placement state.
     */
    inline constexpr uint32_t kPlanFlagCurrentBatchLLEP = 1u << 1;
} // namespace llaminar2::moe_rebalance_abi

namespace llaminar2
{
    /**
     * @brief Exact host action authorized by an authenticated HIP ticket.
     *
     * `None` is a real state rather than a fallback: the complete serial
     * decode graph already acknowledged a non-due boundary, so the host must
     * not launch another graph.  `Acknowledge` retires a pending non-due MTP
     * publication, while `Maintain` enters the collective-bearing graph.
     */
    enum class DeviceMoERebalanceDispatchAction : uint8_t
    {
        Invalid = 0,
        None,
        Acknowledge,
        Maintain,
    };

    /**
     * @brief Immutable host-scheduler view of one device-owned MoE boundary.
     *
     * HIP graph replay has no conditional-node equivalent to CUDA's native
     * `IF(maintenance_due)` transaction.  A rank-local scheduler therefore
     * needs one narrow decision record from every participant before it may
     * submit a collective-bearing maintenance graph.  This record is the only
     * intermediate D2H payload admitted by that policy: it contains lifecycle
     * identity and cadence state, never histograms, commands, expert weights,
     * routing tables, or mutable inference state.
     *
     * The ticket is initialized on the explicit request-reset stream and is
     * subsequently republished by a captured device kernel.  The host may
     * authenticate and compare it, but there is intentionally no API that
     * uploads a host-authored ticket to the device.
     */
    struct DeviceMoERebalanceDispatchTicket
    {
        static constexpr uint32_t kMagic = 0x4d4f4554u; // "MOET"
        static constexpr uint32_t kABIVersion = 1u;

        uint32_t magic = 0;
        uint32_t abi_version = 0;
        uint32_t session_epoch_low = 0;
        uint32_t session_epoch_high = 0;
        uint32_t workspace_generation_low = 0;
        uint32_t workspace_generation_high = 0;
        uint32_t participant_id = 0;
        uint32_t participant_count = 0;
        uint32_t healthy = 0;
        uint32_t controller_version = 0;
        uint32_t decode_rounds_committed = 0;
        uint32_t decode_rounds_until_maintenance = 0;
        uint32_t maintenance_due = 0;
        uint32_t decode_boundary_advanced = 0;
        uint32_t error_code = 0;

        /** @brief Reconstruct the request epoch without host-layout aliases. */
        constexpr uint64_t sessionEpoch() const noexcept
        {
            return static_cast<uint64_t>(session_epoch_low) |
                   (static_cast<uint64_t>(session_epoch_high) << 32u);
        }

        /** @brief Reconstruct the arena generation carried by the ticket. */
        constexpr uint64_t workspaceGeneration() const noexcept
        {
            return static_cast<uint64_t>(workspace_generation_low) |
                   (static_cast<uint64_t>(workspace_generation_high) << 32u);
        }

        /** @brief Check the fixed wire-format identity only. */
        constexpr bool hasValidABI() const noexcept
        {
            return magic == kMagic && abi_version == kABIVersion;
        }

        /**
         * @brief Decode cadence words into the sole legal scheduler action.
         *
         * This method intentionally ignores request identity and health; call
         * @ref matchesLifecycle before acting on its result.  Keeping cadence
         * decoding here prevents orchestration code from growing a second,
         * subtly different truth table.
         */
        constexpr DeviceMoERebalanceDispatchAction dispatchAction()
            const noexcept
        {
            if (maintenance_due == 1u &&
                decode_boundary_advanced == 1u &&
                decode_rounds_until_maintenance == 0u)
            {
                return DeviceMoERebalanceDispatchAction::Maintain;
            }
            if (maintenance_due == 0u &&
                decode_rounds_until_maintenance > 0u)
            {
                if (decode_boundary_advanced == 1u)
                    return DeviceMoERebalanceDispatchAction::Acknowledge;
                if (decode_boundary_advanced == 0u)
                    return DeviceMoERebalanceDispatchAction::None;
            }
            return DeviceMoERebalanceDispatchAction::Invalid;
        }

        /**
         * @brief Authenticate this snapshot against its exact request owner.
         *
         * A due boundary always leaves the once-only advanced bit set and owns
         * a zero remaining budget.  A non-due snapshot owns a positive budget
         * and may be either pending acknowledgement (`advanced=1`) or already
         * acknowledged (`advanced=0`).  The latter is important when HIP's
         * immutable upper-bound schedule deliberately observes early: serial
         * decode embeds publish/ack in its complete graph, so an observation
         * can legitimately find an idle controller rather than a pending
         * boundary.  Encoding all three valid states here prevents malformed
         * bytes from becoming a graph-launch decision without mistaking that
         * acknowledged idle state for corruption.
         */
        constexpr bool matchesLifecycle(
            uint64_t expected_session_epoch,
            uint64_t expected_workspace_generation,
            uint32_t expected_participant_id,
            uint32_t expected_participant_count) const noexcept
        {
            return hasValidABI() &&
                   sessionEpoch() == expected_session_epoch &&
                   workspaceGeneration() == expected_workspace_generation &&
                   participant_id == expected_participant_id &&
                   participant_count == expected_participant_count &&
                   participant_count > 0u &&
                   participant_id < participant_count &&
                   healthy <= 1u &&
                   controller_version == moe_rebalance_abi::kVersion &&
                   dispatchAction() !=
                       DeviceMoERebalanceDispatchAction::Invalid;
        }

        /** @brief Compare fields that must agree across mirrored participants. */
        constexpr bool hasSameDispatchDecision(
            const DeviceMoERebalanceDispatchTicket &other) const noexcept
        {
            return sessionEpoch() == other.sessionEpoch() &&
                   workspaceGeneration() == other.workspaceGeneration() &&
                   participant_count == other.participant_count &&
                   healthy == other.healthy &&
                   controller_version == other.controller_version &&
                   decode_rounds_committed ==
                       other.decode_rounds_committed &&
                   decode_rounds_until_maintenance ==
                       other.decode_rounds_until_maintenance &&
                   maintenance_due == other.maintenance_due &&
                   decode_boundary_advanced ==
                       other.decode_boundary_advanced &&
                   error_code == other.error_code;
        }
    };

    static_assert(
        sizeof(DeviceMoERebalanceDispatchTicket) == 15u * sizeof(uint32_t));
    static_assert(
        std::is_trivially_copyable_v<DeviceMoERebalanceDispatchTicket>);
} // namespace llaminar2
