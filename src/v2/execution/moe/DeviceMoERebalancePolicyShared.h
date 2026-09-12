/**
 * @file DeviceMoERebalancePolicyShared.h
 * @brief Host/device shared MoE rebalance policy helpers.
 *
 * This header owns the small policy decisions that must stay identical across
 * the CPU mirror, CUDA kernels, and ROCm kernels. Keep it dependency-light so it
 * can be included from normal C++, nvcc, and hipcc translation units.
 */

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define LLAMINAR_MOE_REBALANCE_HD __host__ __device__ __forceinline__
#elif defined(__HIPCC__)
#define LLAMINAR_MOE_REBALANCE_HD __host__ __device__ inline __attribute__((always_inline))
#else
#define LLAMINAR_MOE_REBALANCE_HD inline
#endif

namespace llaminar2::moe_rebalance_policy
{
    constexpr uint32_t kPlanExpertPayloadArrival = 1u;
    constexpr uint32_t kPlanResidentExpertAssignment = 2u;
    constexpr uint32_t kPlanOwnershipTransfer = 3u;
    constexpr uint32_t kMaxPolicyParticipants = 8u;
    /** Mathematical floor of a max/min ratio expressed per mille (1.0). */
    constexpr uint32_t kMinimumDynamicImbalanceThresholdPerMille = 1000u;
    constexpr uint32_t kDefaultDynamicImbalanceThresholdPerMille = 1300u;
    constexpr uint32_t kDefaultDynamicMinImprovementPerMille = 50u;
    /**
     * Maximum paired ownership swaps considered for one routed layer.
     *
     * Four gives the device authority enough candidates to address both tier
     * residency and participant skew without letting one layer monopolize a
     * five-cycle physical wave. Keep this policy default independent of the
     * separately tunable physical transfer-slot capacity.
     */
    constexpr uint32_t kDefaultDynamicMaxSwapsPerLayer = 4u;
    /**
     * Maximum device-authored command records retained for one wave.
     *
     * A paired ownership swap can publish two records. Sixteen therefore
     * leaves room for the ordinary four-swap search plus cross-tier arrival
     * and ownership records while keeping the captured command bank bounded.
     */
    constexpr uint32_t kDefaultDynamicMaxPlanEntriesPerWave = 16u;
    constexpr uint64_t kDefaultDynamicMinWindowActivations = 64u;
    /**
     * Expected routed-token lifetime used by the migration payoff test.
     *
     * This is deliberately finite. A very long horizon can make individually
     * plausible cycles look profitable long after the layout has converged,
     * causing continuous background traffic to erase the service-time gain.
     */
    constexpr uint64_t kDefaultMigrationPayoffHorizonTokens = 2048u;
    /**
     * Portable physical migration-cycle capacity retained by default.
     *
     * Exact-geometry shadow arenas are shared across routed layers, so this is
     * a participant-wide concurrency ceiling rather than a per-layer memory
     * multiplier. Five lanes preserved useful four-cycle waves on the
     * Qwen-122B ROCm/CPU production topology while avoiding the residency loss
     * of wider arenas. Deployments may still override the value explicitly.
     */
    constexpr uint32_t kDefaultMigrationTransferSlots = 5u;
    /**
     * Portable default for the per-GPU background migration stream pool.
     *
     * Migration cycle slots retain independent storage, events, and command
     * identity. They do not require one expensive runtime queue each: a small
     * pool can enqueue every slot asynchronously and lets the GPU's copy and
     * compute engines schedule the actual overlap. Four preserves concurrent
     * upload, download, and repack work on current CUDA/ROCm devices while the
     * public runtime setting remains available for topology-specific tuning.
     */
    constexpr uint32_t kDefaultMigrationExecutionStreams = 4u;
    constexpr uint32_t kDefaultDeviceMinLoadSpreadImprovementDivisor = 15u;

    /** Objective preferred for the next slot in one bounded Dynamic wave. */
    enum class DynamicPlacementAxisObjective : uint8_t
    {
        Any,                  ///< Preserve ordinary measured-economy ordering.
        TierResidency,        ///< Prefer promotion/demotion across priorities.
        ParticipantPlacement, ///< Prefer skew reduction inside one priority.
    };

    /**
     * @brief Device-owned coverage state for Dynamic's independent objectives.
     *
     * A bounded wave can otherwise spend every slot on higher-valued tier
     * exchanges and indefinitely starve an economical same-tier correction.
     * Integer priority remains the primary objective: when at least two slots
     * are available, the first slot prefers tier residency and the next slot
     * prefers whichever independent objective is still missing.  A combined
     * cycle satisfies both.  If no candidate advances the preferred objective,
     * callers retain their best ordinary economy candidate as a fallback.
     *
     * This state is shared by the CPU policy oracle and CUDA/HIP policy kernel;
     * it contains no host-owned inference state.
     */
    struct DynamicPlacementAxisProgress
    {
        bool tier_residency = false;        ///< This wave advanced priority.
        bool participant_placement = false; ///< This wave reduced in-tier skew.

        /**
         * @brief Select the objective for the next bounded cycle slot.
         * @param maximum_cycles Total configured cycle slots in this wave.
         * @param accepted_cycles Slots already committed by the authority.
         * @return Preferred objective, or Any when no reservation is useful.
         */
        LLAMINAR_MOE_REBALANCE_HD DynamicPlacementAxisObjective nextObjective(
            uint32_t maximum_cycles,
            uint32_t accepted_cycles) const noexcept
        {
            const uint32_t remaining = maximum_cycles > accepted_cycles
                                           ? maximum_cycles - accepted_cycles
                                           : 0u;
            if (remaining == 0u)
                return DynamicPlacementAxisObjective::Any;
            if (tier_residency && !participant_placement)
                return DynamicPlacementAxisObjective::ParticipantPlacement;
            if (!tier_residency && participant_placement)
                return DynamicPlacementAxisObjective::TierResidency;
            if (!tier_residency && !participant_placement && remaining >= 2u)
                return DynamicPlacementAxisObjective::TierResidency;
            return DynamicPlacementAxisObjective::Any;
        }

        /**
         * @brief Test whether one candidate advances the preferred objective.
         * @param objective Selection returned by @ref nextObjective.
         * @param advances_tier Candidate reduces integer-priority cost.
         * @param advances_participant Candidate reduces in-tier makespan.
         */
        LLAMINAR_MOE_REBALANCE_HD static bool matches(
            DynamicPlacementAxisObjective objective,
            bool advances_tier,
            bool advances_participant) noexcept
        {
            switch (objective)
            {
            case DynamicPlacementAxisObjective::Any:
                return true;
            case DynamicPlacementAxisObjective::TierResidency:
                return advances_tier;
            case DynamicPlacementAxisObjective::ParticipantPlacement:
                return advances_participant;
            }
            return false;
        }

        /**
         * @brief Decide whether axis search must continue at the next layer.
         * @param objective Preferred objective for the current cycle slot.
         * @param found_preferred Whether this layer exposed a matching cycle.
         * @param layer_offset Zero-based position in the current cursor scan.
         * @param layer_count Total number of layers in the scan.
         * @return True when ordinary economy fallback must remain deferred.
         *
         * Axis reservation is wave-wide rather than layer-local.  Spending a
         * reserved slot on the first layer's fallback before inspecting later
         * layers can permanently starve a profitable independent objective.
         * The final layer may use its ordinary fallback when the complete scan
         * proves that no later layer can satisfy the reservation.
         */
        LLAMINAR_MOE_REBALANCE_HD static bool searchesLaterLayerBeforeFallback(
            DynamicPlacementAxisObjective objective,
            bool found_preferred,
            uint32_t layer_offset,
            uint32_t layer_count) noexcept
        {
            return objective != DynamicPlacementAxisObjective::Any &&
                   !found_preferred && layer_offset + 1u < layer_count;
        }

        /**
         * @brief Retain objectives advanced by one accepted cycle.
         * @param advances_tier Whether priority cost decreased.
         * @param advances_participant Whether same-priority makespan decreased.
         */
        LLAMINAR_MOE_REBALANCE_HD void observe(
            bool advances_tier,
            bool advances_participant) noexcept
        {
            tier_residency = tier_residency || advances_tier;
            participant_placement =
                participant_placement || advances_participant;
        }
    };

    /**
     * @brief Bit layout for one all-gathered rebalance-state word.
     *
     * Routing counts and physical placement evidence must describe the same
     * controller epoch. Packing them into one fixed-width collective record
     * makes that relationship structural: the planner cannot accidentally
     * combine current routing counts with stale host-side slot occupancy.
     *
     * Forty-eight count bits are intentionally generous. Even at top-k 16 the
     * counter can represent more than seventeen trillion routed tokens before
     * saturation, while the remaining bits carry the participant-wide active
     * slot count and this expert's exact local storage classification.
     */
    constexpr uint64_t kCollectedStateActivationCountMask =
        (1ULL << 48u) - 1ULL;
    constexpr uint32_t kCollectedStateActiveSlotCountShift = 48u;
    constexpr uint64_t kCollectedStateActiveSlotCountMask =
        0x1ffULL << kCollectedStateActiveSlotCountShift;
    /**
     * @brief This participant is the durable logical owner of the expert.
     *
     * Physical residency alone is insufficient for topology-wide planning:
     * an LLEP arrival or a retained replica can make several participants
     * resident at once.  Keeping the owner bit in the same release-published
     * word as the histogram and residency evidence prevents the global device
     * controller from composing a new epoch from mismatched observations.
     */
    constexpr uint64_t kCollectedStateAuthoritativeOwnerBit = 1ULL << 61u;
    constexpr uint64_t kCollectedStateTransferBackedBit = 1ULL << 62u;
    constexpr uint64_t kCollectedStatePhysicallyResidentBit = 1ULL << 63u;
    constexpr float kDefaultDynamicImbalanceThresholdRatio =
        static_cast<float>(kDefaultDynamicImbalanceThresholdPerMille) / 1000.0f;
    constexpr float kDefaultDynamicMinImprovementRatio =
        static_cast<float>(kDefaultDynamicMinImprovementPerMille) / 1000.0f;

    /**
     * @brief Pack routing and physical storage evidence for one expert.
     *
     * @param activation_count Exact participant-local routing count.
     * @param active_transfer_slots Number of transfer-backed experts currently
     *        live across this participant's complete managed runtime.
     * @param physically_resident Whether this participant can source the bytes.
     * @param transfer_backed Whether the live descriptor consumes transfer storage.
     * @param authoritative_owner Whether this participant owns durable routing
     *        for the expert in the sampled placement epoch.
     * @return One collective word consumed by all backend planners.
     */
    LLAMINAR_MOE_REBALANCE_HD uint64_t packCollectedState(
        uint64_t activation_count,
        uint32_t active_transfer_slots,
        bool physically_resident,
        bool transfer_backed,
        bool authoritative_owner = false) noexcept
    {
        uint64_t packed =
            activation_count & kCollectedStateActivationCountMask;
        packed |=
            (static_cast<uint64_t>(active_transfer_slots > 0x1ffu
                                       ? 0x1ffu
                                       : active_transfer_slots)
             << kCollectedStateActiveSlotCountShift);
        if (physically_resident)
            packed |= kCollectedStatePhysicallyResidentBit;
        if (transfer_backed)
            packed |= kCollectedStateTransferBackedBit;
        if (authoritative_owner)
            packed |= kCollectedStateAuthoritativeOwnerBit;
        return packed;
    }

    /// Return the routing-count component of an all-gathered state word.
    LLAMINAR_MOE_REBALANCE_HD uint64_t collectedStateActivationCount(
        uint64_t packed) noexcept
    {
        return packed & kCollectedStateActivationCountMask;
    }

    /// Return the participant-wide active transfer-slot count.
    LLAMINAR_MOE_REBALANCE_HD uint32_t collectedStateActiveTransferSlots(
        uint64_t packed) noexcept
    {
        return static_cast<uint32_t>(
            (packed & kCollectedStateActiveSlotCountMask) >>
            kCollectedStateActiveSlotCountShift);
    }

    /// Return whether this participant physically stores the expert payload.
    LLAMINAR_MOE_REBALANCE_HD bool collectedStatePhysicallyResident(
        uint64_t packed) noexcept
    {
        return (packed & kCollectedStatePhysicallyResidentBit) != 0ULL;
    }

    /// Return whether this participant's expert payload uses transfer storage.
    LLAMINAR_MOE_REBALANCE_HD bool collectedStateTransferBacked(
        uint64_t packed) noexcept
    {
        return (packed & kCollectedStateTransferBackedBit) != 0ULL;
    }

    /// Return whether this participant is the durable logical expert owner.
    LLAMINAR_MOE_REBALANCE_HD bool collectedStateAuthoritativeOwner(
        uint64_t packed) noexcept
    {
        return (packed & kCollectedStateAuthoritativeOwnerBit) != 0ULL;
    }

    struct LoadSpreadDelta
    {
        uint64_t current_total = 0;
        uint64_t current_min = 0;
        uint64_t current_max = 0;
        uint64_t proposed_total = 0;
        uint64_t proposed_min = 0;
        uint64_t proposed_max = 0;
        uint64_t current_spread = 0;
        uint64_t proposed_spread = 0;
        uint64_t improvement = 0;
        uint64_t required_improvement = 0;
        bool total_preserved = false;
        bool improves = false;
        bool meets_floor = false;
    };

    struct DestinationChoice
    {
        uint32_t destination_participant = 0;
        uint32_t proposed_resident_mask = 0;
        uint32_t source_participant = 0;
        uint64_t projected_shift = 0;
        LoadSpreadDelta delta{};
        bool valid = false;
    };

    struct OwnershipSwapChoice
    {
        uint32_t overloaded_participant = 0;
        uint32_t underloaded_participant = 0;
        uint32_t heavy_expert = 0;
        uint32_t light_expert = 0;
        uint64_t heavy_count = 0;
        uint64_t light_count = 0;
        uint64_t old_min_load = 0;
        uint64_t old_max_load = 0;
        uint64_t new_min_load = 0;
        uint64_t new_max_load = 0;
        uint64_t improvement = 0;
        bool valid = false;
    };

    /**
     * @brief Evidence scope that qualifies one Dynamic ownership decision.
     *
     * `routed_activations` names the complete routed layer window before any
     * tier or participant partitioning.  The optimization may inspect only a
     * subset of those routes, but partitioning must not silently redefine the
     * public minimum-window policy.  Keeping both values in one typed argument
     * makes host, CUDA, and ROCm callers state that accounting explicitly.
     */
    struct DynamicOwnershipEvidenceWindow
    {
        uint64_t routed_activations = 0;
        uint64_t minimum_routed_activations =
            kDefaultDynamicMinWindowActivations;

        /** @return Whether the complete routed window clears its policy floor. */
        LLAMINAR_MOE_REBALANCE_HD bool sufficient() const noexcept
        {
            return routed_activations >= minimum_routed_activations;
        }
    };

    /**
     * @brief Build whole-window evidence when the candidate spans every owner.
     *
     * Single-tier and legacy Dynamic callers optimize over the complete owner
     * set, so summing participant loads recovers the routed layer window. A
     * multi-tier authority instead supplies its pre-partition layer total
     * directly and must not call this helper on one tier slice.
     *
     * @param participant_load Routed load for every participant in the window.
     * @param participant_count Number of participant load entries.
     * @param minimum_routed_activations Configured evidence floor.
     * @return Saturation-safe typed evidence for the shared selector.
     */
    LLAMINAR_MOE_REBALANCE_HD DynamicOwnershipEvidenceWindow
    dynamicOwnershipEvidenceFromParticipantLoads(
        const uint64_t *participant_load,
        uint32_t participant_count,
        uint64_t minimum_routed_activations) noexcept
    {
        DynamicOwnershipEvidenceWindow evidence{
            0u,
            minimum_routed_activations,
        };
        if (!participant_load)
            return evidence;
        for (uint32_t participant = 0;
             participant < participant_count;
             ++participant)
        {
            const uint64_t load = participant_load[participant];
            evidence.routed_activations =
                UINT64_MAX - evidence.routed_activations < load
                    ? UINT64_MAX
                    : evidence.routed_activations + load;
        }
        return evidence;
    }

    /**
     * @brief Storage-lifetime classification for one stable GPU transfer slot.
     *
     * Runtime placement keeps storage residency and current compute assignment
     * as separate concerns. An expert can own or retain resident bytes in a
     * transfer slot even when `local_compute_mask` is zero because the current
     * least-loaded assignment sends all of its rows elsewhere. Slot allocation
     * must therefore derive liveness from ownership, residency, and the next
     * assignment instead of using compute eligibility as a storage lease.
     */
    struct TransferSlotOccupancy
    {
        bool occupied = false;
        bool protected_from_reuse = false;
    };

    /**
     * @brief Machine-readable reasons that a live transfer-slot claim is invalid.
     *
     * A claim audit is part of the transfer state machine, not merely a logging
     * aid. Keeping its reason bits in this host/device shared policy guarantees
     * that CPU diagnostics, CUDA maintenance graphs, and ROCm maintenance graphs
     * classify the same runtime publication identically.
     */
    enum TransferSlotClaimInvalidReason : uint32_t
    {
        TransferSlotClaimValid = 0u,
        TransferSlotClaimMissingValidFlag = 1u << 0u,
        TransferSlotClaimMissingResidentFlag = 1u << 1u,
        TransferSlotClaimMissingLocalResidency = 1u << 2u,
        TransferSlotClaimNegativeSlot = 1u << 3u,
        TransferSlotClaimExceedsCompileTimeCapacity = 1u << 4u,
        TransferSlotClaimExceedsDirectoryCapacity = 1u << 5u,
        TransferSlotClaimDirectoryIdentityMismatch = 1u << 6u,
        TransferSlotClaimOccupantMismatch = 1u << 7u,
    };

    /**
     * @brief Classification of one runtime descriptor's physical slot claim.
     */
    struct TransferSlotClaim
    {
        bool claims_storage = false;
        uint32_t invalid_reasons = TransferSlotClaimValid;

        LLAMINAR_MOE_REBALANCE_HD bool valid() const noexcept
        {
            return claims_storage &&
                   invalid_reasons == TransferSlotClaimValid;
        }
    };

    /**
     * @brief Classify one descriptor using the canonical physical-liveness rule.
     *
     * Current row assignment is intentionally absent from this API. An
     * authoritative owner can receive zero rows while its transfer allocation
     * remains the only durable copy. Conversely, a stale TransferSlot bit on a
     * remote, nonresident descriptor does not consume local storage. All three
     * backends must use this helper so claim accounting cannot drift by backend.
     *
     * @param descriptor_flags Runtime expert descriptor flags.
     * @param resident_mask Domain-wide physical residency mask.
     * @param local_participant_bit Bit identifying this participant.
     * @param owner_participant Authoritative participant in the descriptor.
     * @param local_participant This runtime table's participant.
     * @param local_slot Stable transfer-directory slot, or a negative sentinel.
     * @param compile_time_capacity Maximum slot ID representable by the ABI.
     * @param directory_capacity Runtime addressable directory capacity.
     * @param valid_flag Descriptor flag proving the payload descriptor is valid.
     * @param resident_flag Descriptor flag proving local payload residency.
     * @param transfer_slot_flag Descriptor flag identifying transfer storage.
     */
    LLAMINAR_MOE_REBALANCE_HD TransferSlotClaim classifyTransferSlotClaim(
        uint32_t descriptor_flags,
        uint32_t resident_mask,
        uint32_t local_participant_bit,
        int32_t owner_participant,
        uint32_t local_participant,
        int32_t local_slot,
        uint32_t compile_time_capacity,
        uint32_t directory_capacity,
        uint32_t valid_flag,
        uint32_t resident_flag,
        uint32_t transfer_slot_flag) noexcept
    {
        TransferSlotClaim claim;
        const bool local_resident =
            (resident_mask & local_participant_bit) != 0u;
        const bool authoritative_local_owner =
            owner_participant == static_cast<int32_t>(local_participant);
        claim.claims_storage =
            (descriptor_flags & transfer_slot_flag) != 0u &&
            (local_resident || authoritative_local_owner);
        if (!claim.claims_storage)
            return claim;

        if ((descriptor_flags & valid_flag) == 0u)
            claim.invalid_reasons |= TransferSlotClaimMissingValidFlag;
        if ((descriptor_flags & resident_flag) == 0u)
            claim.invalid_reasons |= TransferSlotClaimMissingResidentFlag;
        if (!local_resident)
            claim.invalid_reasons |= TransferSlotClaimMissingLocalResidency;
        if (local_slot < 0)
        {
            claim.invalid_reasons |= TransferSlotClaimNegativeSlot;
            return claim;
        }

        const uint32_t slot = static_cast<uint32_t>(local_slot);
        if (slot >= compile_time_capacity)
            claim.invalid_reasons |=
                TransferSlotClaimExceedsCompileTimeCapacity;
        if (slot >= directory_capacity)
            claim.invalid_reasons |=
                TransferSlotClaimExceedsDirectoryCapacity;
        return claim;
    }

    /**
     * @brief Retire every local pointer and flag for a departed GPU payload.
     *
     * Local residency, pointer-bearing matrix descriptors, the stable slot ID,
     * and execution flags form one publication. Clearing only LocalCompute leaves
     * a descriptor that can later be mistaken for durable storage. This helper
     * makes retirement a single host/device operation used by CUDA and ROCm.
     *
     * The logical expert ID, authoritative owner, and policy-only flags are
     * preserved because remote placement metadata remains meaningful after the
     * local bytes depart.
     *
     * @param descriptor Runtime expert descriptor to retire.
     * @param resident_mask Domain-wide resident mask to update.
     * @param local_participant_bit Bit identifying this participant.
     * @param local_payload_flags Valid/resident/compute/replica/transfer flags.
     */
    template <typename ExpertDescriptor>
    LLAMINAR_MOE_REBALANCE_HD void retireLocalPayloadPublication(
        ExpertDescriptor &descriptor,
        uint32_t &resident_mask,
        uint32_t local_participant_bit,
        uint32_t local_payload_flags) noexcept
    {
        resident_mask &= ~local_participant_bit;
        descriptor.gate = decltype(descriptor.gate){};
        descriptor.up = decltype(descriptor.up){};
        descriptor.down = decltype(descriptor.down){};
        descriptor.local_slot = -1;
        descriptor.flags &= ~local_payload_flags;
    }

    /**
     * @brief Classify whether a transfer-backed expert occupies a local slot.
     *
     * An authoritative local owner is always protected, including while it has
     * no rows in the current LLEP assignment. A non-owner resident replica
     * occupies the slot but remains replaceable when the next assignment does
     * not read it locally. An assignment-local expert is protected even if its
     * resident bit is awaiting publication in the current transfer wave.
     *
     * @param transfer_backed       Whether the descriptor names a transfer slot.
     * @param assigned_locally      Whether the next assignment reads the expert locally.
     * @param resident_mask         Participants currently storing the expert.
     * @param local_participant_bit Bit naming the local participant.
     * @param owner_participant     Authoritative participant in the descriptor.
     * @param local_participant     Participant materializing destination slots.
     * @return Independent occupied and protected-from-reuse decisions.
     */
    LLAMINAR_MOE_REBALANCE_HD TransferSlotOccupancy classifyTransferSlotOccupancy(
        bool transfer_backed,
        bool assigned_locally,
        uint32_t resident_mask,
        uint32_t local_participant_bit,
        int32_t owner_participant,
        uint32_t local_participant) noexcept
    {
        TransferSlotOccupancy result{};
        if (!transfer_backed)
            return result;

        const bool owner_local =
            owner_participant == static_cast<int32_t>(local_participant);
        const bool locally_resident =
            (resident_mask & local_participant_bit) != 0u;

        result.occupied =
            owner_local || locally_resident || assigned_locally;
        result.protected_from_reuse =
            result.occupied && (owner_local || assigned_locally);
        return result;
    }

    /**
     * @brief Decide whether a prefill assignment still reads a slot occupant.
     *
     * Transfer-backed LLEP prefill is a forward-only streaming pipeline. Before
     * layer `N` leases physical payload storage, its transfer stream waits on an
     * event recorded after all compute-stream work through layer `N - 1`.
     * Consequently, a non-owner replica from another layer is no longer read by
     * the current forward pass and may be replaced transactionally. A replica
     * from the layer being prepared remains protected when that layer's
     * device-owned assignment sends rows to the local participant.
     *
     * Keeping this rule backend-neutral prevents CUDA and ROCm from drifting on
     * transfer-directory lifetime. Ownership remains an independent protection
     * in classifyTransferSlotOccupancy(); this helper only answers whether the
     * current layer assignment contributes a live read.
     *
     * @param prepared_layer              Layer whose arrivals are being leased.
     * @param occupant_layer              Layer currently naming the physical slot.
     * @param same_layer_assigned_locally Whether that occupant has local rows in
     *                                    the prepared layer's assignment.
     * @return true only when the current layer will read the existing occupant.
     */
    LLAMINAR_MOE_REBALANCE_HD bool prefillAssignmentReadsTransferSlotOccupant(
        uint32_t prepared_layer,
        uint32_t occupant_layer,
        bool same_layer_assigned_locally) noexcept
    {
        return prepared_layer == occupant_layer &&
               same_layer_assigned_locally;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t participantBit(uint32_t participant) noexcept
    {
        return 1u << participant;
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t directedParticipantEdgeBit(
        uint32_t source_participant,
        uint32_t destination_participant,
        uint32_t participant_stride) noexcept
    {
        if (participant_stride == 0u ||
            source_participant >= participant_stride ||
            destination_participant >= participant_stride)
        {
            return 0ULL;
        }
        const uint32_t bit_index =
            source_participant * participant_stride + destination_participant;
        return bit_index < 64u ? (1ULL << bit_index) : 0ULL;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t validParticipantMask(uint32_t participant_count) noexcept
    {
        return (1u << participant_count) - 1u;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t residentCount(
        uint32_t resident_mask,
        uint32_t participant_count) noexcept
    {
        uint32_t count = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if ((resident_mask & participantBit(participant)) != 0u)
                ++count;
        }
        return count;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t payloadBucketSlots(
        uint32_t requested_slots,
        uint32_t max_slot_capacity) noexcept
    {
        if (requested_slots == 0 || max_slot_capacity == 0)
            return 0;

        uint32_t bucket = 1u;
        while (bucket < requested_slots && bucket < max_slot_capacity)
        {
            const uint32_t next = bucket << 1u;
            if (next <= bucket)
                break;
            bucket = next;
        }
        return bucket > max_slot_capacity ? max_slot_capacity : bucket;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t payloadBucketIndex(
        uint32_t bucket_slots) noexcept
    {
        uint32_t index = 0;
        while (bucket_slots > 1u)
        {
            bucket_slots >>= 1u;
            ++index;
        }
        return index;
    }

    LLAMINAR_MOE_REBALANCE_HD bool ratioAtLeastPerMille(
        uint64_t numerator,
        uint64_t denominator,
        uint32_t threshold_per_mille) noexcept
    {
        if (denominator == 0u)
            return numerator > 0u;
        if (threshold_per_mille == 0u)
            return true;
        if (numerator > UINT64_MAX / 1000ULL)
            return true;
        return numerator * 1000ULL >=
               denominator * static_cast<uint64_t>(threshold_per_mille);
    }

    /**
     * @brief Test a strict scalar reduction against an integer economy floor.
     *
     * The quotient/remainder form avoids overflowing either side of a
     * cross-multiplied per-mille comparison. CUDA, HIP, and the CPU oracle use
     * this exact helper so a borderline candidate cannot be accepted by one
     * controller backend and rejected by another.
     *
     * @param before Objective value before the candidate cycle.
     * @param after Objective value after the candidate cycle.
     * @param minimum_reduction_per_mille Required fractional reduction.
     * @return True only when the objective strictly decreases by the floor.
     */
    LLAMINAR_MOE_REBALANCE_HD bool relativeReductionAtLeastPerMille(
        uint64_t before,
        uint64_t after,
        uint32_t minimum_reduction_per_mille) noexcept
    {
        if (before == 0u || after >= before)
            return false;
        if (minimum_reduction_per_mille == 0u)
            return true;
        if (minimum_reduction_per_mille >= 1000u)
            return after == 0u;

        const uint64_t whole =
            (before / 1000u) * minimum_reduction_per_mille;
        const uint64_t remainder_product =
            (before % 1000u) * minimum_reduction_per_mille;
        const uint64_t rounded_remainder =
            (remainder_product + 999u) / 1000u;
        return before - after >= whole + rounded_remainder;
    }

    /**
     * @brief Reject an economy candidate that regresses any measured phase.
     *
     * Decode and prefill are independent production objectives. Summing their
     * service costs before admission would let a large win in one phase hide a
     * regression in the other. The aggregate payoff gate still decides whether
     * the complete cycle is worth its movement cost, while this helper enforces
     * the orthogonal invariant that no measured phase becomes slower. Equality
     * is intentional: a cycle may improve one phase while leaving another
     * phase unchanged, and several such cycles can form one useful wave.
     *
     * @param before Exact critical-path service cost before movement by phase.
     * @param after Exact critical-path service cost after movement by phase.
     * @param phase_count Number of valid entries in both arrays.
     * @return True when every measured phase is unchanged or improved.
     */
    LLAMINAR_MOE_REBALANCE_HD bool phaseObjectivesDoNotRegress(
        const uint64_t *before,
        const uint64_t *after,
        uint32_t phase_count) noexcept
    {
        if (!before || !after || phase_count == 0u)
            return false;
        for (uint32_t phase = 0u; phase < phase_count; ++phase)
        {
            if (after[phase] > before[phase])
                return false;
        }
        return true;
    }

    /**
     * @brief Apply the Dynamic planner's lexicographic improvement floor.
     *
     * Cross-tier placement cost is authoritative. Same-priority makespan is
     * considered only when the priority cost is unchanged, preserving the
     * hottest-experts-first contract while rejecting tiny reshuffles that
     * cannot plausibly repay physical movement.
     */
    LLAMINAR_MOE_REBALANCE_HD bool dynamicObjectiveImprovesByPerMille(
        uint64_t current_priority_cost,
        uint64_t current_same_priority_makespan,
        uint64_t candidate_priority_cost,
        uint64_t candidate_same_priority_makespan,
        uint32_t minimum_improvement_per_mille) noexcept
    {
        if (candidate_priority_cost < current_priority_cost)
        {
            return relativeReductionAtLeastPerMille(
                current_priority_cost,
                candidate_priority_cost,
                minimum_improvement_per_mille);
        }
        return candidate_priority_cost == current_priority_cost &&
               relativeReductionAtLeastPerMille(
                   current_same_priority_makespan,
                   candidate_same_priority_makespan,
                   minimum_improvement_per_mille);
    }

    /**
     * @brief Test whether one equal-priority participant set is skewed enough.
     *
     * @param participant_load Exact routed load under the current owner map.
     * @param participant_priority Opaque integer priority per participant.
     * @param participant_count Number of valid entries in both arrays.
     * @param selected_priority Priority shared by the candidate cycle.
     * @param threshold_per_mille Required maximum/minimum load ratio.
     */
    LLAMINAR_MOE_REBALANCE_HD bool samePriorityLoadIsImbalanced(
        const uint64_t *participant_load,
        const int32_t *participant_priority,
        uint32_t participant_count,
        int32_t selected_priority,
        uint32_t threshold_per_mille) noexcept
    {
        if (!participant_load || !participant_priority ||
            participant_count < 2u)
        {
            return false;
        }
        uint32_t members = 0u;
        uint64_t minimum = UINT64_MAX;
        uint64_t maximum = 0u;
        for (uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            if (participant_priority[participant] != selected_priority)
                continue;
            const uint64_t load = participant_load[participant];
            minimum = load < minimum ? load : minimum;
            maximum = load > maximum ? load : maximum;
            ++members;
        }
        return members >= 2u && maximum != 0u &&
               ratioAtLeastPerMille(
                   maximum,
                   minimum == UINT64_MAX ? 0u : minimum,
                   threshold_per_mille);
    }

    LLAMINAR_MOE_REBALANCE_HD bool finiteRatioImprovesByPerMille(
        uint64_t old_max,
        uint64_t old_min,
        uint64_t new_max,
        uint64_t new_min,
        uint32_t min_improvement_per_mille) noexcept
    {
        if (old_min == 0u || new_min == 0u || old_max == 0u)
            return false;
        if (min_improvement_per_mille == 0u)
            return new_max * old_min < old_max * new_min;

        const uint64_t keep_per_mille =
            min_improvement_per_mille >= 1000u
                ? 0ULL
                : 1000ULL - static_cast<uint64_t>(min_improvement_per_mille);
        if (new_max != 0u && old_min > UINT64_MAX / new_max)
            return false;
        const uint64_t lhs0 = new_max * old_min;
        if (lhs0 > UINT64_MAX / 1000ULL)
            return false;
        const uint64_t lhs = lhs0 * 1000ULL;

        if (old_max != 0u && new_min > UINT64_MAX / old_max)
            return true;
        const uint64_t rhs0 = old_max * new_min;
        if (keep_per_mille != 0u && rhs0 > UINT64_MAX / keep_per_mille)
            return true;
        const uint64_t rhs = rhs0 * keep_per_mille;
        return lhs <= rhs;
    }

    /**
     * @brief Test whether one ownership move preserves durable slot capacity.
     *
     * Ownership and physical storage are intentionally separate from the
     * current compute assignment. Moving an expert whose source bytes live in
     * static model storage consumes one new durable transfer slot at the
     * destination. Moving a transfer-backed expert releases one durable slot
     * at the source and consumes one at the destination. The destination may
     * already have transfer-backed storage only when a prior transaction left
     * a physical replica there, in which case the move does not grow its
     * working set.
     *
     * Callers pass transaction-local occupancy arrays. After accepting a move,
     * they must call @ref applyOwnershipTransferOccupancy before evaluating the
     * next candidate. This makes a multi-command captured wave obey the same
     * bounded-storage invariant as its eventual runtime publication.
     *
     * @param participant_active_transfer_slots Current durable claims per participant.
     * @param expert_transfer_backed_participant_mask Physical transfer-backed copies.
     * @param expert Logical expert moved by the command.
     * @param source_participant Current authoritative source participant.
     * @param destination_participant New authoritative destination participant.
     * @param participant_count Number of participants in the placement domain.
     * @param active_transfer_slot_capacity Durable slots available per participant.
     * @return true only when the resulting destination occupancy is representable.
     */
    LLAMINAR_MOE_REBALANCE_HD bool ownershipTransferFitsPersistentCapacity(
        const uint32_t *participant_active_transfer_slots,
        const uint32_t *expert_transfer_backed_participant_mask,
        uint32_t expert,
        uint32_t source_participant,
        uint32_t destination_participant,
        uint32_t participant_count,
        uint32_t active_transfer_slot_capacity) noexcept
    {
        if (!participant_active_transfer_slots ||
            !expert_transfer_backed_participant_mask ||
            source_participant >= participant_count ||
            destination_participant >= participant_count ||
            source_participant == destination_participant ||
            active_transfer_slot_capacity == UINT32_MAX)
        {
            return active_transfer_slot_capacity == UINT32_MAX &&
                   source_participant < participant_count &&
                   destination_participant < participant_count &&
                   source_participant != destination_participant;
        }

        for (uint32_t participant = 0;
             participant < participant_count;
             ++participant)
        {
            if (participant_active_transfer_slots[participant] >
                active_transfer_slot_capacity)
            {
                return false;
            }
        }

        const uint32_t transfer_backed_mask =
            expert_transfer_backed_participant_mask[expert];
        const bool source_transfer_backed =
            (transfer_backed_mask & participantBit(source_participant)) != 0u;
        if (source_transfer_backed &&
            participant_active_transfer_slots[source_participant] == 0u)
        {
            return false;
        }
        const bool destination_already_transfer_backed =
            (transfer_backed_mask & participantBit(destination_participant)) != 0u;
        const uint32_t destination_growth =
            destination_already_transfer_backed ? 0u : 1u;
        if (destination_growth > active_transfer_slot_capacity)
            return false;
        return participant_active_transfer_slots[destination_participant] <=
               active_transfer_slot_capacity - destination_growth;
    }

    /**
     * @brief Advance transaction-local storage ownership after one accepted move.
     *
     * The helper mutates only planner scratch. Runtime descriptors and transfer
     * directory entries remain unchanged until the graph-ordered unpack/apply
     * phases authenticate and publish the command.
     *
     * @return false when the supplied state cannot describe the requested move.
     */
    LLAMINAR_MOE_REBALANCE_HD bool applyOwnershipTransferOccupancy(
        uint32_t *participant_active_transfer_slots,
        uint32_t *expert_transfer_backed_participant_mask,
        uint32_t expert,
        uint32_t source_participant,
        uint32_t destination_participant) noexcept
    {
        if (!participant_active_transfer_slots ||
            !expert_transfer_backed_participant_mask ||
            source_participant == destination_participant)
        {
            return false;
        }

        const uint32_t source_bit = participantBit(source_participant);
        const uint32_t destination_bit = participantBit(destination_participant);
        uint32_t &mask = expert_transfer_backed_participant_mask[expert];
        if ((mask & source_bit) != 0u)
        {
            if (participant_active_transfer_slots[source_participant] == 0u)
                return false;
            --participant_active_transfer_slots[source_participant];
            mask &= ~source_bit;
        }
        if ((mask & destination_bit) == 0u)
        {
            ++participant_active_transfer_slots[destination_participant];
            mask |= destination_bit;
        }
        return true;
    }

    /**
     * @brief Find the best capacity-preserving expert-owner exchange.
     *
     * The most-loaded and least-loaded participants define the skew that must
     * be reduced.  The selected payload pair is nevertheless searched
     * exhaustively: exchanging the hottest expert with the coldest expert can
     * overshoot the balance point and make the ratio worse even when another
     * pair is profitable.  This exact search is deterministic and shared by
     * host, CUDA, and ROCm controllers so every authority publishes the same
     * ownership transition from identical histogram evidence.
     *
     * Candidates minimize the resulting parallel makespan first, maximize the
     * resulting minimum load second, avoid reversing the overloaded endpoint
     * when those objectives tie, and finally use expert IDs for a stable tie
     * break.  Transfer-slot constraints are applied during enumeration, not
     * after selecting an otherwise impossible mathematical winner.
     *
     * @param participant_load Aggregate routed activations per participant.
     * @param expert_counts Routed activations per expert.
     * @param expert_owner Participant-local owner index per expert.
     * @param num_experts Number of entries in both expert arrays.
     * @param participant_count Number of entries in participant_load.
     * @param evidence Complete routed-layer evidence before placement
     *        partitioning.
     * @param imbalance_threshold_per_mille Minimum current max/min ratio.
     * @param min_improvement_per_mille Required ratio improvement.
     * @param expert_transfer_backed_participant_mask Optional physical slots.
     * @param participant_active_transfer_slots Optional live slot counts.
     * @param active_transfer_slot_capacity Optional per-participant capacity.
     * @return The best executable paired swap, or an invalid choice.
     */
    LLAMINAR_MOE_REBALANCE_HD OwnershipSwapChoice bestDynamicOwnershipSwap(
        const uint64_t *participant_load,
        const uint64_t *expert_counts,
        const int32_t *expert_owner,
        uint32_t num_experts,
        uint32_t participant_count,
        DynamicOwnershipEvidenceWindow evidence,
        uint32_t imbalance_threshold_per_mille,
        uint32_t min_improvement_per_mille,
        const uint32_t *expert_transfer_backed_participant_mask = nullptr,
        const uint32_t *participant_active_transfer_slots = nullptr,
        uint32_t active_transfer_slot_capacity = UINT32_MAX) noexcept
    {
        OwnershipSwapChoice choice{};
        if (!participant_load ||
            !expert_counts ||
            !expert_owner ||
            num_experts == 0u ||
            participant_count < 2u)
        {
            return choice;
        }

        const bool has_transfer_masks =
            expert_transfer_backed_participant_mask != nullptr;
        const bool has_transfer_counts =
            participant_active_transfer_slots != nullptr;
        const bool has_transfer_capacity =
            active_transfer_slot_capacity != UINT32_MAX;
        const bool enforce_transfer_capacity =
            has_transfer_masks && has_transfer_counts &&
            has_transfer_capacity;
        if ((has_transfer_masks || has_transfer_counts ||
             has_transfer_capacity) &&
            !enforce_transfer_capacity)
        {
            return choice;
        }
        if (enforce_transfer_capacity &&
            participant_count > kMaxPolicyParticipants)
        {
            return choice;
        }

        uint32_t overloaded = 0;
        uint32_t underloaded = 0;
        uint64_t max_load = 0;
        uint64_t min_load = UINT64_MAX;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t load = participant_load[participant];
            if (load > max_load)
            {
                max_load = load;
                overloaded = participant;
            }
            if (load < min_load)
            {
                min_load = load;
                underloaded = participant;
            }
        }
        if (!evidence.sufficient() ||
            max_load == 0u ||
            overloaded == underloaded)
        {
            return choice;
        }
        if (min_load != 0u &&
            !ratioAtLeastPerMille(max_load, min_load, imbalance_threshold_per_mille))
        {
            return choice;
        }

        if (enforce_transfer_capacity)
        {
            for (uint32_t participant = 0;
                 participant < participant_count;
                 ++participant)
            {
                /*
                 * A count above the declared active capacity means the
                 * runtime has already violated its storage contract. Refusing
                 * to publish another command keeps that invariant fail-closed.
                 */
                if (participant_active_transfer_slots[participant] >
                    active_transfer_slot_capacity)
                {
                    return choice;
                }
            }
        }

        /*
         * Each ownership swap sends one expert in both directions. A
         * participant already at capacity can accept its incoming payload only
         * when its outgoing expert is transfer-backed and therefore releases a
         * durable slot in this exact wave. Filtering candidates here lets the
         * policy select the best physically executable pair instead of
         * publishing an impossible mathematical winner.
         */
        const bool overloaded_requires_transfer_departure =
            enforce_transfer_capacity &&
            participant_active_transfer_slots[overloaded] >=
                active_transfer_slot_capacity;
        const bool underloaded_requires_transfer_departure =
            enforce_transfer_capacity &&
            participant_active_transfer_slots[underloaded] >=
                active_transfer_slot_capacity;

        for (uint32_t heavy_expert = 0;
             heavy_expert < num_experts;
             ++heavy_expert)
        {
            if (expert_owner[heavy_expert] !=
                static_cast<int32_t>(overloaded))
            {
                continue;
            }
            if (overloaded_requires_transfer_departure &&
                (expert_transfer_backed_participant_mask[heavy_expert] &
                 participantBit(overloaded)) == 0u)
            {
                continue;
            }
            const uint64_t heavy_count = expert_counts[heavy_expert];

            for (uint32_t light_expert = 0;
                 light_expert < num_experts;
                 ++light_expert)
            {
                if (expert_owner[light_expert] !=
                    static_cast<int32_t>(underloaded))
                {
                    continue;
                }
                if (underloaded_requires_transfer_departure &&
                    (expert_transfer_backed_participant_mask[light_expert] &
                     participantBit(underloaded)) == 0u)
                {
                    continue;
                }
                const uint64_t light_count = expert_counts[light_expert];
                if (heavy_count <= light_count)
                    continue;

                const uint64_t proposed_overloaded_load =
                    max_load - heavy_count + light_count;
                const uint64_t proposed_underloaded_load =
                    min_load - light_count + heavy_count;
                uint64_t new_min = UINT64_MAX;
                uint64_t new_max = 0u;
                for (uint32_t participant = 0;
                     participant < participant_count;
                     ++participant)
                {
                    uint64_t load = participant_load[participant];
                    if (participant == overloaded)
                        load = proposed_overloaded_load;
                    else if (participant == underloaded)
                        load = proposed_underloaded_load;
                    if (load < new_min)
                        new_min = load;
                    if (load > new_max)
                        new_max = load;
                }

                const bool accept =
                    min_load > 0u
                        ? finiteRatioImprovesByPerMille(
                              max_load,
                              min_load,
                              new_max,
                              new_min,
                              min_improvement_per_mille)
                        : (new_min > 0u || new_max < max_load);
                if (!accept)
                    continue;

                const bool reverses_extrema =
                    proposed_overloaded_load < proposed_underloaded_load;
                const uint64_t current_overloaded_load =
                    choice.valid
                        ? max_load - choice.heavy_count + choice.light_count
                        : 0u;
                const uint64_t current_underloaded_load =
                    choice.valid
                        ? min_load - choice.light_count + choice.heavy_count
                        : 0u;
                const bool current_reverses_extrema =
                    choice.valid &&
                    current_overloaded_load < current_underloaded_load;
                const bool better =
                    !choice.valid ||
                    new_max < choice.new_max_load ||
                    (new_max == choice.new_max_load &&
                     new_min > choice.new_min_load) ||
                    (new_max == choice.new_max_load &&
                     new_min == choice.new_min_load &&
                     reverses_extrema != current_reverses_extrema &&
                     !reverses_extrema) ||
                    (new_max == choice.new_max_load &&
                     new_min == choice.new_min_load &&
                     reverses_extrema == current_reverses_extrema &&
                     (heavy_expert < choice.heavy_expert ||
                      (heavy_expert == choice.heavy_expert &&
                       light_expert < choice.light_expert)));
                if (!better)
                    continue;

                choice.overloaded_participant = overloaded;
                choice.underloaded_participant = underloaded;
                choice.heavy_expert = heavy_expert;
                choice.light_expert = light_expert;
                choice.heavy_count = heavy_count;
                choice.light_count = light_count;
                choice.old_min_load = min_load;
                choice.old_max_load = max_load;
                choice.new_min_load = new_min;
                choice.new_max_load = new_max;
                const uint64_t old_spread = max_load - min_load;
                const uint64_t new_spread = new_max - new_min;
                choice.improvement = old_spread > new_spread
                                         ? old_spread - new_spread
                                         : 1u;
                choice.valid = true;
            }
        }
        return choice;
    }

    /**
     * @brief Advance transfer-slot occupancy after an accepted ownership swap.
     *
     * This updates the controller's transaction-local model only; publication
     * still occurs later through the normal copy/apply lifecycle. Keeping the
     * accounting helper shared prevents CUDA and ROCm from selecting different
     * second swaps within the same captured wave.
     */
    LLAMINAR_MOE_REBALANCE_HD bool applyDynamicOwnershipSwapTransferOccupancy(
        uint32_t *participant_active_transfer_slots,
        uint32_t *expert_transfer_backed_participant_mask,
        const OwnershipSwapChoice &choice) noexcept
    {
        if (!participant_active_transfer_slots ||
            !expert_transfer_backed_participant_mask ||
            !choice.valid)
        {
            return false;
        }

        return applyOwnershipTransferOccupancy(
                   participant_active_transfer_slots,
                   expert_transfer_backed_participant_mask,
                   choice.heavy_expert,
                   choice.overloaded_participant,
                   choice.underloaded_participant) &&
               applyOwnershipTransferOccupancy(
                   participant_active_transfer_slots,
                   expert_transfer_backed_participant_mask,
                   choice.light_expert,
                   choice.underloaded_participant,
                   choice.overloaded_participant);
    }

    LLAMINAR_MOE_REBALANCE_HD bool applyDynamicOwnershipSwap(
        uint64_t *participant_load,
        int32_t *expert_owner,
        const OwnershipSwapChoice &choice) noexcept
    {
        if (!participant_load || !expert_owner || !choice.valid)
            return false;
        participant_load[choice.overloaded_participant] =
            choice.old_max_load - choice.heavy_count + choice.light_count;
        participant_load[choice.underloaded_participant] =
            choice.old_min_load - choice.light_count + choice.heavy_count;
        expert_owner[choice.heavy_expert] =
            static_cast<int32_t>(choice.underloaded_participant);
        expert_owner[choice.light_expert] =
            static_cast<int32_t>(choice.overloaded_participant);
        return true;
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveMeetsSpreadImprovementFloor(
        uint64_t accepted_spread_improvement,
        uint32_t requested_payload_slots,
        uint32_t min_spread_improvement_per_payload_slot) noexcept
    {
        if (requested_payload_slots == 0u ||
            min_spread_improvement_per_payload_slot == 0u)
        {
            return true;
        }
        return accepted_spread_improvement >=
               static_cast<uint64_t>(requested_payload_slots) *
                   static_cast<uint64_t>(min_spread_improvement_per_payload_slot);
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveMeetsRealizedRouterBenefitFloor(
        uint64_t realized_router_spread_improvement,
        uint32_t requested_payload_slots,
        uint32_t min_router_spread_improvement_per_payload_slot,
        bool existing_hot_cache_active) noexcept
    {
        if (!existing_hot_cache_active ||
            requested_payload_slots == 0u ||
            min_router_spread_improvement_per_payload_slot == 0u)
        {
            return true;
        }
        return realized_router_spread_improvement >=
               static_cast<uint64_t>(requested_payload_slots) *
                   static_cast<uint64_t>(min_router_spread_improvement_per_payload_slot);
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveMeetsPostLoadSpreadCeiling(
        uint64_t post_policy_load_spread,
        uint64_t post_policy_load_total,
        uint32_t requested_payload_slots,
        uint32_t max_post_load_spread_per_mille) noexcept
    {
        if (requested_payload_slots == 0u ||
            max_post_load_spread_per_mille == 0u ||
            post_policy_load_total == 0u)
        {
            return true;
        }
        if (post_policy_load_spread > UINT64_MAX / 1000ULL)
            return false;
        const uint64_t lhs = post_policy_load_spread * 1000ULL;
        // A mathematical RHS larger than uint64 cannot reject a representable
        // LHS. Do not wrap it into a small threshold on a long demand horizon.
        if (post_policy_load_total > UINT64_MAX / max_post_load_spread_per_mille)
            return true;
        const uint64_t rhs =
            post_policy_load_total *
            static_cast<uint64_t>(max_post_load_spread_per_mille);
        return lhs <= rhs;
    }

    LLAMINAR_MOE_REBALANCE_HD bool transferWaveImprovesAggregateLoadSpread(
        uint64_t pre_wave_load_spread,
        uint64_t post_wave_load_spread,
        uint64_t pre_wave_load_total,
        uint64_t post_wave_load_total,
        uint32_t requested_payload_slots) noexcept
    {
        if (requested_payload_slots == 0u)
            return true;
        if (pre_wave_load_total == 0u ||
            post_wave_load_total == 0u ||
            pre_wave_load_total != post_wave_load_total)
        {
            return false;
        }
        return post_wave_load_spread < pre_wave_load_spread;
    }

    /**
     * @brief Decide whether the whole-wave participant totals permit a paid wave.
     *
     * Participant totals are a useful guard, but they are not the execution
     * critical path: opposite skews in different serial layers can cancel in
     * the totals.  A separately proven critical-path payoff therefore permits
     * a tied or worse aggregate total.  Callers must still apply the stricter
     * per-layer aggregate-spread and configured post-wave ceiling gates.
     *
     * @param pre_policy_load_spread Participant-total spread before the wave.
     * @param post_policy_load_spread Participant-total spread after the wave.
     * @param pre_policy_load_total Total routed work before the wave.
     * @param post_policy_load_total Total routed work after the wave.
     * @param requested_payload_slots Paid payload slots required by the wave.
     * @param realized_critical_path_payback Whether an independent critical-
     *        path measure proves the wave useful.
     * @return True when participant totals do not veto the paid wave.
     */
    LLAMINAR_MOE_REBALANCE_HD bool transferWaveParticipantSpreadIsAcceptable(
        uint64_t pre_policy_load_spread,
        uint64_t post_policy_load_spread,
        uint64_t pre_policy_load_total,
        uint64_t post_policy_load_total,
        uint32_t requested_payload_slots,
        bool realized_critical_path_payback) noexcept
    {
        if (requested_payload_slots == 0u)
            return true;
        if (pre_policy_load_total == 0u ||
            post_policy_load_total == 0u ||
            pre_policy_load_total != post_policy_load_total)
        {
            return false;
        }
        return post_policy_load_spread < pre_policy_load_spread ||
               realized_critical_path_payback;
    }

    template <typename PlanEntry>
    LLAMINAR_MOE_REBALANCE_HD uint32_t prunePayloadArrivalsPreservingResidentAssignments(
        PlanEntry *entries,
        uint32_t command_count,
        uint32_t *resident_command_count = nullptr) noexcept
    {
        uint32_t write_index = 0;
        uint32_t residents = 0;
        if (entries != nullptr)
        {
            for (uint32_t read_index = 0; read_index < command_count; ++read_index)
            {
                const PlanEntry entry = entries[read_index];
                if (entry.op != kPlanResidentExpertAssignment)
                    continue;
                entries[write_index++] = entry;
                ++residents;
            }
        }
        if (resident_command_count != nullptr)
            *resident_command_count = residents;
        return write_index;
    }

    LLAMINAR_MOE_REBALANCE_HD uint32_t residentOrdinal(
        uint32_t resident_mask,
        uint32_t participant_count,
        uint32_t target_participant) noexcept
    {
        uint32_t ordinal = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if ((resident_mask & participantBit(participant)) == 0u)
                continue;
            if (participant == target_participant)
                return ordinal;
            ++ordinal;
        }
        return participant_count;
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t projectedParticipantLoadForExpert(
        uint64_t expert_count,
        uint32_t resident_mask,
        uint32_t participant_count,
        uint32_t participant) noexcept
    {
        if (expert_count == 0 ||
            participant >= participant_count ||
            (resident_mask & participantBit(participant)) == 0u)
        {
            return 0;
        }
        const uint32_t count = residentCount(resident_mask, participant_count);
        if (count == 0)
            return 0;
        const uint64_t base = expert_count / static_cast<uint64_t>(count);
        const uint64_t remainder = expert_count % static_cast<uint64_t>(count);
        const uint32_t ordinal = residentOrdinal(resident_mask, participant_count, participant);
        return base + (ordinal < remainder ? 1u : 0u);
    }

    LLAMINAR_MOE_REBALANCE_HD bool addExpertLoadLeastLoaded(
        uint64_t *participant_load,
        uint64_t expert_count,
        uint32_t resident_mask,
        uint32_t participant_count) noexcept
    {
        if (!participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants)
        {
            return false;
        }

        resident_mask &= validParticipantMask(participant_count);
        if (resident_mask == 0u)
            return false;

        uint64_t remaining = expert_count;
        while (remaining > 0ULL)
        {
            uint64_t min_load = UINT64_MAX;
            uint64_t next_load = UINT64_MAX;
            uint32_t min_count = 0;
            for (uint32_t participant = 0; participant < participant_count; ++participant)
            {
                if ((resident_mask & participantBit(participant)) == 0u)
                    continue;
                const uint64_t load = participant_load[participant];
                if (load < min_load)
                {
                    next_load = min_load;
                    min_load = load;
                    min_count = 1u;
                }
                else if (load == min_load)
                {
                    ++min_count;
                }
                else if (load < next_load)
                {
                    next_load = load;
                }
            }

            if (min_count == 0u)
                return false;

            if (next_load != UINT64_MAX && next_load > min_load)
            {
                const uint64_t delta_to_next = next_load - min_load;
                if (delta_to_next <= UINT64_MAX / static_cast<uint64_t>(min_count))
                {
                    const uint64_t rows_to_next =
                        delta_to_next * static_cast<uint64_t>(min_count);
                    if (rows_to_next <= remaining)
                    {
                        for (uint32_t participant = 0; participant < participant_count; ++participant)
                        {
                            if ((resident_mask & participantBit(participant)) != 0u &&
                                participant_load[participant] == min_load)
                            {
                                participant_load[participant] += delta_to_next;
                            }
                        }
                        remaining -= rows_to_next;
                        continue;
                    }
                }
            }

            const uint64_t rows_per_min = remaining / static_cast<uint64_t>(min_count);
            uint64_t extra_rows = remaining % static_cast<uint64_t>(min_count);
            for (uint32_t participant = 0; participant < participant_count; ++participant)
            {
                if ((resident_mask & participantBit(participant)) == 0u ||
                    participant_load[participant] != min_load)
                {
                    continue;
                }
                participant_load[participant] +=
                    rows_per_min + (extra_rows > 0ULL ? 1ULL : 0ULL);
                if (extra_rows > 0ULL)
                    --extra_rows;
            }
            remaining = 0ULL;
        }

        return true;
    }

    LLAMINAR_MOE_REBALANCE_HD void finalizeLoadSpread(
        const uint64_t *participant_load,
        uint32_t participant_count,
        uint64_t &total,
        uint64_t &min_load,
        uint64_t &max_load) noexcept
    {
        total = 0;
        min_load = UINT64_MAX;
        max_load = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t load = participant_load[participant];
            total += load;
            if (load < min_load)
                min_load = load;
            if (load > max_load)
                max_load = load;
        }
        if (participant_count == 0 || min_load == UINT64_MAX)
            min_load = 0;
    }

    /**
     * @brief Project one layer-local ownership swap onto the whole wave load.
     *
     * A Dynamic candidate is selected from one layer, but the devices execute
     * every layer in the maintenance wave. Independently beneficial layer
     * swaps can all move work in the same direction and make the aggregate
     * participant imbalance worse. This helper evaluates the candidate against
     * the transaction's current all-layer load before any command is published.
     *
     * The returned proposal preserves the input array. Call
     * @ref applyDynamicOwnershipSwapToAggregateLoadIfImproved only after all
     * physical-source and transfer-capacity checks have also passed.
     *
     * @param aggregate_participant_load Current projected load across the wave.
     * @param participant_count Number of participants represented by the array.
     * @param choice Layer-local paired ownership swap under consideration.
     * @return Exact before/after spread evidence; `improves` is true only for a
     *         total-preserving strict aggregate improvement.
     */
    LLAMINAR_MOE_REBALANCE_HD LoadSpreadDelta
    evaluateDynamicOwnershipSwapAgainstAggregateLoad(
        const uint64_t *aggregate_participant_load,
        uint32_t participant_count,
        const OwnershipSwapChoice &choice) noexcept
    {
        LoadSpreadDelta delta{};
        if (!aggregate_participant_load ||
            !choice.valid ||
            participant_count < 2u ||
            participant_count > kMaxPolicyParticipants ||
            choice.overloaded_participant >= participant_count ||
            choice.underloaded_participant >= participant_count ||
            choice.overloaded_participant == choice.underloaded_participant ||
            choice.heavy_count <= choice.light_count)
        {
            return delta;
        }

        finalizeLoadSpread(
            aggregate_participant_load,
            participant_count,
            delta.current_total,
            delta.current_min,
            delta.current_max);
        delta.current_spread = delta.current_max - delta.current_min;

        const uint64_t shifted_load =
            choice.heavy_count - choice.light_count;
        const uint64_t source_load =
            aggregate_participant_load[choice.overloaded_participant];
        const uint64_t destination_load =
            aggregate_participant_load[choice.underloaded_participant];
        if (source_load < shifted_load ||
            destination_load > UINT64_MAX - shifted_load)
        {
            return delta;
        }

        delta.proposed_min = UINT64_MAX;
        for (uint32_t participant = 0;
             participant < participant_count;
             ++participant)
        {
            uint64_t proposed_load = aggregate_participant_load[participant];
            if (participant == choice.overloaded_participant)
                proposed_load -= shifted_load;
            else if (participant == choice.underloaded_participant)
                proposed_load += shifted_load;

            delta.proposed_total += proposed_load;
            if (proposed_load < delta.proposed_min)
                delta.proposed_min = proposed_load;
            if (proposed_load > delta.proposed_max)
                delta.proposed_max = proposed_load;
        }
        if (delta.proposed_min == UINT64_MAX)
            delta.proposed_min = 0u;
        delta.proposed_spread =
            delta.proposed_max - delta.proposed_min;
        delta.total_preserved =
            delta.proposed_total == delta.current_total;
        delta.improves =
            delta.total_preserved &&
            delta.proposed_spread < delta.current_spread;
        delta.meets_floor = delta.improves;
        delta.improvement =
            delta.improves
                ? delta.current_spread - delta.proposed_spread
                : 0u;
        return delta;
    }

    /**
     * @brief Commit one layer-local swap to the wave aggregate when beneficial.
     *
     * This mutates planner scratch only. Runtime ownership remains unchanged
     * until the graph-ordered payload publication and bank flip authenticate
     * the complete command wave.
     *
     * @param aggregate_participant_load Mutable all-layer projected load.
     * @param participant_count Number of participants represented by the array.
     * @param choice Layer-local paired ownership swap under consideration.
     * @param out_improvement Optional destination for the aggregate spread gain.
     * @return true only when the aggregate load was updated with a strict gain.
     */
    LLAMINAR_MOE_REBALANCE_HD bool
    applyDynamicOwnershipSwapToAggregateLoadIfImproved(
        uint64_t *aggregate_participant_load,
        uint32_t participant_count,
        const OwnershipSwapChoice &choice,
        uint64_t *out_improvement = nullptr) noexcept
    {
        if (out_improvement)
            *out_improvement = 0u;
        const LoadSpreadDelta delta =
            evaluateDynamicOwnershipSwapAgainstAggregateLoad(
                aggregate_participant_load,
                participant_count,
                choice);
        if (!delta.improves)
            return false;

        const uint64_t shifted_load =
            choice.heavy_count - choice.light_count;
        aggregate_participant_load[choice.overloaded_participant] -=
            shifted_load;
        aggregate_participant_load[choice.underloaded_participant] +=
            shifted_load;
        if (out_improvement)
            *out_improvement = delta.improvement;
        return true;
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t requiredLoadSpreadImprovement(
        uint64_t current_total,
        uint32_t min_improvement,
        uint32_t min_improvement_divisor) noexcept
    {
        uint64_t required_improvement = static_cast<uint64_t>(min_improvement);
        if (min_improvement_divisor > 0)
        {
            const uint64_t relative_floor =
                current_total / static_cast<uint64_t>(min_improvement_divisor);
            if (relative_floor > required_improvement)
                required_improvement = relative_floor;
        }
        return required_improvement;
    }

    LLAMINAR_MOE_REBALANCE_HD bool expertCountCanMeetLoadSpreadFloor(
        uint64_t expert_count,
        uint64_t current_total,
        uint32_t min_improvement,
        uint32_t min_improvement_divisor) noexcept
    {
        const uint64_t required_improvement = requiredLoadSpreadImprovement(
            current_total,
            min_improvement,
            min_improvement_divisor);
        return required_improvement == 0 || expert_count >= required_improvement;
    }

    LLAMINAR_MOE_REBALANCE_HD bool candidateValueIsBetter(
        uint64_t candidate_value,
        uint64_t candidate_count,
        uint32_t candidate_expert,
        uint64_t best_value,
        uint64_t best_count,
        uint32_t best_expert) noexcept
    {
        if (candidate_value == 0)
            return false;
        if (best_value == 0)
            return true;
        if (candidate_value != best_value)
            return candidate_value > best_value;
        if (candidate_count != best_count)
            return candidate_count > best_count;
        return candidate_expert < best_expert;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentCanMeetLoadSpreadFloorByCount(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t min_improvement,
        uint32_t min_improvement_divisor) noexcept
    {
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask)
        {
            return false;
        }

        uint64_t current_total = 0;
        uint64_t current_min = 0;
        uint64_t current_max = 0;
        finalizeLoadSpread(
            current_participant_load,
            participant_count,
            current_total,
            current_min,
            current_max);
        (void)current_min;
        (void)current_max;

        return expertCountCanMeetLoadSpreadFloor(
            expert_count,
            current_total,
            min_improvement,
            min_improvement_divisor);
    }

    LLAMINAR_MOE_REBALANCE_HD LoadSpreadDelta evaluateAddingResidentLoadSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        LoadSpreadDelta delta{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask)
        {
            return delta;
        }

        finalizeLoadSpread(
            current_participant_load,
            participant_count,
            delta.current_total,
            delta.current_min,
            delta.current_max);
        delta.current_spread = delta.current_max - delta.current_min;
        delta.required_improvement = requiredLoadSpreadImprovement(
            delta.current_total,
            min_improvement,
            min_improvement_divisor);

        delta.proposed_total = 0;
        delta.proposed_min = UINT64_MAX;
        delta.proposed_max = 0;
        bool underflow = false;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            const uint64_t new_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    proposed_resident_mask,
                    participant_count,
                    participant);
            if (current_participant_load[participant] < old_contribution)
                underflow = true;
            const uint64_t proposed_load =
                underflow
                    ? 0
                    : current_participant_load[participant] - old_contribution + new_contribution;
            delta.proposed_total += proposed_load;
            if (proposed_load < delta.proposed_min)
                delta.proposed_min = proposed_load;
            if (proposed_load > delta.proposed_max)
                delta.proposed_max = proposed_load;
        }
        if (participant_count == 0 || delta.proposed_min == UINT64_MAX)
            delta.proposed_min = 0;

        delta.total_preserved = !underflow && delta.proposed_total == delta.current_total;
        delta.proposed_spread = delta.proposed_max - delta.proposed_min;
        delta.improves =
            delta.total_preserved && delta.proposed_spread < delta.current_spread;
        delta.improvement =
            delta.improves ? (delta.current_spread - delta.proposed_spread) : 0;
        delta.meets_floor =
            delta.improves && delta.improvement >= delta.required_improvement;
        return delta;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentImprovesLoadSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint64_t *scratch_participant_load,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        if (!current_participant_load ||
            !scratch_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask)
        {
            return false;
        }

        const LoadSpreadDelta delta = evaluateAddingResidentLoadSpread(
            current_participant_load,
            expert_count,
            current_resident_mask,
            proposed_resident_mask,
            participant_count,
            min_improvement,
            min_improvement_divisor);
        if (!delta.meets_floor)
            return false;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            const uint64_t new_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    proposed_resident_mask,
                    participant_count,
                    participant);
            scratch_participant_load[participant] =
                current_participant_load[participant] - old_contribution + new_contribution;
        }
        return true;
    }

    LLAMINAR_MOE_REBALANCE_HD LoadSpreadDelta evaluateAddingResidentLeastLoadedSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        LoadSpreadDelta delta{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants ||
            current_resident_mask == proposed_resident_mask)
        {
            return delta;
        }

        finalizeLoadSpread(
            current_participant_load,
            participant_count,
            delta.current_total,
            delta.current_min,
            delta.current_max);
        delta.current_spread = delta.current_max - delta.current_min;
        delta.required_improvement = requiredLoadSpreadImprovement(
            delta.current_total,
            min_improvement,
            min_improvement_divisor);

        uint64_t proposed_load[kMaxPolicyParticipants] = {};
        bool underflow = false;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            if (current_participant_load[participant] < old_contribution)
                underflow = true;
            proposed_load[participant] =
                underflow ? 0ULL : current_participant_load[participant] - old_contribution;
        }

        if (underflow ||
            !addExpertLoadLeastLoaded(
                proposed_load,
                expert_count,
                proposed_resident_mask,
                participant_count))
        {
            return delta;
        }

        finalizeLoadSpread(
            proposed_load,
            participant_count,
            delta.proposed_total,
            delta.proposed_min,
            delta.proposed_max);
        delta.total_preserved = delta.proposed_total == delta.current_total;
        delta.proposed_spread = delta.proposed_max - delta.proposed_min;
        delta.improves =
            delta.total_preserved && delta.proposed_spread < delta.current_spread;
        delta.improvement =
            delta.improves ? (delta.current_spread - delta.proposed_spread) : 0;
        delta.meets_floor =
            delta.improves && delta.improvement >= delta.required_improvement;
        return delta;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentImprovesLeastLoadedSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint64_t *scratch_participant_load,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        if (!current_participant_load ||
            !scratch_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants ||
            current_resident_mask == proposed_resident_mask)
        {
            return false;
        }

        const LoadSpreadDelta delta = evaluateAddingResidentLeastLoadedSpread(
            current_participant_load,
            expert_count,
            current_resident_mask,
            proposed_resident_mask,
            participant_count,
            min_improvement,
            min_improvement_divisor);
        if (!delta.meets_floor)
            return false;

        bool underflow = false;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint64_t old_contribution =
                projectedParticipantLoadForExpert(
                    expert_count,
                    current_resident_mask,
                    participant_count,
                    participant);
            if (current_participant_load[participant] < old_contribution)
                underflow = true;
            scratch_participant_load[participant] =
                underflow ? 0ULL : current_participant_load[participant] - old_contribution;
        }
        if (underflow)
            return false;

        return addExpertLoadLeastLoaded(
            scratch_participant_load,
            expert_count,
            proposed_resident_mask,
            participant_count);
    }

    LLAMINAR_MOE_REBALANCE_HD uint64_t dynamicMinimumProjectedShift(
        uint32_t window_size_tokens,
        uint64_t current_total,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        uint64_t floor = window_size_tokens > 0u
                             ? static_cast<uint64_t>(window_size_tokens / 16u)
                             : 0ULL;
        if (floor < 2ULL)
            floor = 2ULL;
        const uint64_t configured_floor = requiredLoadSpreadImprovement(
            current_total,
            min_improvement,
            min_improvement_divisor);
        return configured_floor > floor ? configured_floor : floor;
    }

    LLAMINAR_MOE_REBALANCE_HD LoadSpreadDelta evaluateAddingResidentDynamicSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t source_participant,
        uint32_t destination_participant,
        uint32_t window_size_tokens = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        LoadSpreadDelta delta{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            current_resident_mask == proposed_resident_mask ||
            source_participant >= participant_count ||
            destination_participant >= participant_count ||
            source_participant == destination_participant ||
            (current_resident_mask & participantBit(source_participant)) == 0u ||
            (proposed_resident_mask & participantBit(destination_participant)) == 0u)
        {
            return delta;
        }

        finalizeLoadSpread(
            current_participant_load,
            participant_count,
            delta.current_total,
            delta.current_min,
            delta.current_max);
        delta.current_spread = delta.current_max - delta.current_min;
        delta.required_improvement = dynamicMinimumProjectedShift(
            window_size_tokens,
            delta.current_total,
            min_improvement,
            min_improvement_divisor);

        const uint64_t source_load = current_participant_load[source_participant];
        const uint64_t destination_load = current_participant_load[destination_participant];
        if (source_load <= destination_load)
            return delta;

        const uint64_t equalizing_shift = (source_load - destination_load + 1ULL) / 2ULL;
        const uint64_t shift = equalizing_shift < expert_count ? equalizing_shift : expert_count;

        delta.proposed_total = 0;
        delta.proposed_min = UINT64_MAX;
        delta.proposed_max = 0;
        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            uint64_t proposed_load = current_participant_load[participant];
            if (participant == source_participant)
            {
                proposed_load = proposed_load > shift ? proposed_load - shift : 0ULL;
            }
            else if (participant == destination_participant)
            {
                proposed_load += shift;
            }
            delta.proposed_total += proposed_load;
            if (proposed_load < delta.proposed_min)
                delta.proposed_min = proposed_load;
            if (proposed_load > delta.proposed_max)
                delta.proposed_max = proposed_load;
        }
        if (participant_count == 0 || delta.proposed_min == UINT64_MAX)
            delta.proposed_min = 0;

        delta.total_preserved = delta.proposed_total == delta.current_total;
        delta.proposed_spread = delta.proposed_max - delta.proposed_min;
        delta.improves =
            delta.total_preserved && delta.proposed_spread < delta.current_spread;
        delta.improvement = delta.improves ? shift : 0ULL;
        delta.meets_floor = delta.improves && shift >= delta.required_improvement;
        return delta;
    }

    LLAMINAR_MOE_REBALANCE_HD bool addingResidentImprovesDynamicSpread(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t proposed_resident_mask,
        uint32_t participant_count,
        uint32_t source_participant,
        uint32_t destination_participant,
        uint64_t *scratch_participant_load,
        uint32_t window_size_tokens = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        if (!current_participant_load || !scratch_participant_load)
            return false;

        const LoadSpreadDelta delta = evaluateAddingResidentDynamicSpread(
            current_participant_load,
            expert_count,
            current_resident_mask,
            proposed_resident_mask,
            participant_count,
            source_participant,
            destination_participant,
            window_size_tokens,
            min_improvement,
            min_improvement_divisor);
        if (!delta.meets_floor || delta.improvement == 0ULL)
            return false;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            uint64_t proposed_load = current_participant_load[participant];
            if (participant == source_participant)
                proposed_load = proposed_load > delta.improvement ? proposed_load - delta.improvement : 0ULL;
            else if (participant == destination_participant)
                proposed_load += delta.improvement;
            scratch_participant_load[participant] = proposed_load;
        }
        return true;
    }

    LLAMINAR_MOE_REBALANCE_HD bool destinationChoiceIsBetter(
        const DestinationChoice &candidate,
        const DestinationChoice &best,
        const uint64_t *current_participant_load) noexcept
    {
        if (!candidate.valid)
            return false;
        if (!best.valid)
            return true;
        if (candidate.delta.improvement != best.delta.improvement)
            return candidate.delta.improvement > best.delta.improvement;
        if (current_participant_load)
        {
            const uint64_t candidate_load =
                current_participant_load[candidate.destination_participant];
            const uint64_t best_load =
                current_participant_load[best.destination_participant];
            if (candidate_load != best_load)
                return candidate_load < best_load;
        }
        return candidate.destination_participant < best.destination_participant;
    }

    LLAMINAR_MOE_REBALANCE_HD DestinationChoice bestMissingResidentDestination(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t participant_count,
        const uint32_t *planned_replicas_per_participant = nullptr,
        uint32_t max_replicas_per_participant = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        DestinationChoice best{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0)
        {
            return best;
        }

        const uint32_t valid_mask = validParticipantMask(participant_count);
        current_resident_mask &= valid_mask;
        if (current_resident_mask == valid_mask)
            return best;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint32_t bit = participantBit(participant);
            if ((current_resident_mask & bit) != 0u)
                continue;
            if (planned_replicas_per_participant &&
                max_replicas_per_participant > 0u &&
                planned_replicas_per_participant[participant] >= max_replicas_per_participant)
            {
                continue;
            }

            DestinationChoice candidate{};
            candidate.destination_participant = participant;
            candidate.proposed_resident_mask = (current_resident_mask | bit) & valid_mask;
            candidate.delta = evaluateAddingResidentLoadSpread(
                current_participant_load,
                expert_count,
                current_resident_mask,
                candidate.proposed_resident_mask,
                participant_count,
                min_improvement,
                min_improvement_divisor);
            candidate.valid = candidate.delta.meets_floor;
            if (destinationChoiceIsBetter(candidate, best, current_participant_load))
                best = candidate;
        }

        return best;
    }

	    LLAMINAR_MOE_REBALANCE_HD DestinationChoice bestLeastLoadedMissingResidentDestination(
	        const uint64_t *current_participant_load,
	        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t participant_count,
        const uint32_t *planned_replicas_per_participant = nullptr,
        uint32_t max_replicas_per_participant = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        DestinationChoice best{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants)
        {
            return best;
        }

        const uint32_t valid_mask = validParticipantMask(participant_count);
        current_resident_mask &= valid_mask;
        if (current_resident_mask == valid_mask)
            return best;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint32_t bit = participantBit(participant);
            if ((current_resident_mask & bit) != 0u)
                continue;
            if (planned_replicas_per_participant &&
                max_replicas_per_participant > 0u &&
                planned_replicas_per_participant[participant] >= max_replicas_per_participant)
            {
                continue;
            }

            DestinationChoice candidate{};
            candidate.destination_participant = participant;
            candidate.proposed_resident_mask = (current_resident_mask | bit) & valid_mask;
            candidate.delta = evaluateAddingResidentLeastLoadedSpread(
                current_participant_load,
                expert_count,
                current_resident_mask,
                candidate.proposed_resident_mask,
                participant_count,
                min_improvement,
                min_improvement_divisor);
            candidate.valid = candidate.delta.meets_floor;
            if (destinationChoiceIsBetter(candidate, best, current_participant_load))
                best = candidate;
        }

	        return best;
	    }

    LLAMINAR_MOE_REBALANCE_HD int32_t firstResidentParticipant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int32_t preferred_participant,
        int32_t excluded_participant) noexcept;

    LLAMINAR_MOE_REBALANCE_HD DestinationChoice bestDynamicMissingResidentDestination(
        const uint64_t *current_participant_load,
        uint64_t expert_count,
        uint32_t current_resident_mask,
        uint32_t participant_count,
        int32_t preferred_source_participant = -1,
        const uint32_t *planned_replicas_per_participant = nullptr,
        uint32_t max_replicas_per_participant = 0,
        uint32_t window_size_tokens = 0,
        uint32_t min_improvement = 0,
        uint32_t min_improvement_divisor = 0) noexcept
    {
        DestinationChoice best{};
        if (!current_participant_load ||
            expert_count == 0 ||
            participant_count == 0 ||
            participant_count > kMaxPolicyParticipants)
        {
            return best;
        }

        const uint32_t valid_mask = validParticipantMask(participant_count);
        current_resident_mask &= valid_mask;
        if (current_resident_mask == 0u || current_resident_mask == valid_mask)
            return best;

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            const uint32_t bit = participantBit(participant);
            if ((current_resident_mask & bit) != 0u)
                continue;
            if (planned_replicas_per_participant &&
                max_replicas_per_participant > 0u &&
                planned_replicas_per_participant[participant] >= max_replicas_per_participant)
            {
                continue;
            }

            const int32_t source = firstResidentParticipant(
                current_resident_mask,
                participant_count,
                preferred_source_participant,
                static_cast<int32_t>(participant));
            if (source < 0)
                continue;

            DestinationChoice candidate{};
            candidate.destination_participant = participant;
            candidate.source_participant = static_cast<uint32_t>(source);
            candidate.proposed_resident_mask = (current_resident_mask | bit) & valid_mask;
            candidate.delta = evaluateAddingResidentDynamicSpread(
                current_participant_load,
                expert_count,
                current_resident_mask,
                candidate.proposed_resident_mask,
                participant_count,
                candidate.source_participant,
                candidate.destination_participant,
                window_size_tokens,
                min_improvement,
                min_improvement_divisor);
            candidate.projected_shift = candidate.delta.improvement;
            candidate.valid = candidate.delta.meets_floor;
            if (destinationChoiceIsBetter(candidate, best, current_participant_load))
                best = candidate;
        }

        return best;
    }

    LLAMINAR_MOE_REBALANCE_HD bool commandBufferFull(
        uint32_t command_count,
        uint32_t command_capacity) noexcept
    {
        return command_count >= command_capacity;
    }

    /**
     * @brief Typed admission result for one atomic Dynamic ownership swap.
     *
     * A swap contributes two inseparable commands. Reaching the configured
     * wave budget is ordinary scheduling backpressure: the remaining
     * candidates belong to a later epoch. Exhausting the larger physical plan
     * before that declared budget is an invariant violation. Keeping these
     * outcomes distinct prevents an exactly-full, valid wave from being
     * reported as command-buffer corruption.
     */
    enum class DynamicPlanAdmission : uint8_t
    {
        Admit = 0,
        ConfiguredWaveLimit,
        PhysicalPlanOverflow,
    };

    /**
     * @brief Classify whether an atomic Dynamic command group can be appended.
     *
     * The policy limit is evaluated first because it may intentionally equal
     * the physical capacity. A zero policy limit means that physical capacity
     * is the only bound. Subtraction-based checks avoid unsigned overflow.
     *
     * @param command_count Commands already published in the current wave.
     * @param required_entries Atomic commands required by the candidate.
     * @param configured_wave_limit User/policy limit, or zero for unbounded.
     * @param physical_plan_capacity Allocated command-buffer capacity.
     * @return The exact admission or rejection reason.
     */
    LLAMINAR_MOE_REBALANCE_HD DynamicPlanAdmission dynamicPlanAdmission(
        uint32_t command_count,
        uint32_t required_entries,
        uint32_t configured_wave_limit,
        uint32_t physical_plan_capacity) noexcept
    {
        if (configured_wave_limit != 0u &&
            (command_count > configured_wave_limit ||
             required_entries > configured_wave_limit - command_count))
        {
            return DynamicPlanAdmission::ConfiguredWaveLimit;
        }
        if (command_count > physical_plan_capacity ||
            required_entries > physical_plan_capacity - command_count)
        {
            return DynamicPlanAdmission::PhysicalPlanOverflow;
        }
        return DynamicPlanAdmission::Admit;
    }

    template <typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool hasValidRootParticipant(
        const RebalanceConfig &config) noexcept
    {
        return config.root_participant < config.participant_count;
    }

    template <typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool isRootParticipant(
        const RebalanceConfig &config) noexcept
    {
        return config.participant_id == config.root_participant;
    }

    LLAMINAR_MOE_REBALANCE_HD int32_t firstResidentParticipant(
        uint32_t resident_mask,
        uint32_t participant_count,
        int32_t preferred_participant = -1,
        int32_t excluded_participant = -1) noexcept
    {
        if (preferred_participant >= 0 &&
            preferred_participant < static_cast<int32_t>(participant_count) &&
            preferred_participant != excluded_participant &&
            (resident_mask & participantBit(static_cast<uint32_t>(preferred_participant))) != 0u)
        {
            return preferred_participant;
        }

        for (uint32_t participant = 0; participant < participant_count; ++participant)
        {
            if (static_cast<int32_t>(participant) == excluded_participant)
                continue;
            if ((resident_mask & participantBit(participant)) != 0u)
                return static_cast<int32_t>(participant);
        }
        return -1;
    }

    template <typename ExpertDescriptor, typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool candidateCanAffectLocalCompute(
        const ExpertDescriptor &desc,
        uint32_t resident_mask,
        const RebalanceConfig &config,
        bool plan_missing_arrivals) noexcept
    {
        const uint32_t participant_bit = participantBit(config.participant_id);
        const bool local_resident = (resident_mask & participant_bit) != 0u;
        const bool owner_local =
            desc.owner_participant == static_cast<int32_t>(config.participant_id);
        const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
        if (replicated && local_resident)
            return true;
        if (!plan_missing_arrivals || local_resident || owner_local)
            return false;
        return firstResidentParticipant(
                   resident_mask,
                   config.participant_count,
                   desc.owner_participant,
                   static_cast<int32_t>(config.participant_id)) >= 0;
    }

    template <typename ExpertDescriptor, typename RebalanceConfig>
    LLAMINAR_MOE_REBALANCE_HD bool candidateCanAffectDomainCompute(
        const ExpertDescriptor &desc,
        uint32_t resident_mask,
        const RebalanceConfig &config,
        bool plan_missing_arrivals) noexcept
    {
        const uint32_t valid_mask = validParticipantMask(config.participant_count);
        resident_mask &= valid_mask;

        /*
         * Domain-root planning is allowed to prefer the static owner when a
         * resident source must be selected, but it must not convert ownership
         * into residency.  The GPU payload packer can only copy bytes from a
         * participant whose bit is already present in the runtime resident mask.
         */
        const bool replicated = (resident_mask & (resident_mask - 1u)) != 0u;
        if (replicated)
            return true;
        if (!plan_missing_arrivals || resident_mask == 0u || resident_mask == valid_mask)
            return false;
        return firstResidentParticipant(
                   resident_mask,
                   config.participant_count,
                   desc.owner_participant,
                   -1) >= 0;
    }
} // namespace llaminar2::moe_rebalance_policy

#undef LLAMINAR_MOE_REBALANCE_HD
