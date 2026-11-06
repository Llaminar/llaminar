/**
 * @file MoEOverlayNativeInvocationABI.h
 * @brief Native metadata for the original bounded inference-publication slots.
 *
 * One original protocol slot owns one immutable admission and two endpoint
 * packet-retirement receipts. All learned graph families borrow that same
 * roster; family channel identity selects the matching immutable admission.
 * Payload banks and graph topology remain owned by their original family.
 * Receipt publication follows the final payload kernel, rather than the earlier
 * semantic Complete marker. Full graph/arena retirement remains separate.
 */
#pragma once
#include "MoEOverlayActivationEpochABI.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_NATIVE_INVOCATION_HD __host__ __device__
#if defined(__HIPCC__)
#define LLAMINAR_MOE_NATIVE_INVOCATION_INLINE inline __attribute__((always_inline))
#else
#define LLAMINAR_MOE_NATIVE_INVOCATION_INLINE __forceinline__
#endif
#else
#define LLAMINAR_MOE_NATIVE_INVOCATION_HD
#define LLAMINAR_MOE_NATIVE_INVOCATION_INLINE inline
#endif

namespace llaminar2
{
    /** Cold-selected original admission owner; no automatic alternate path. */
    enum class MoEOverlayActivationAdmissionPolicy : std::uint32_t
    {
        HostOrderedEpoch = 0, ///< Existing explicit host/remote transaction boundary.
        OriginalNativeSlots = 1, ///< Same original slots, consumed and handed off on device.
    };
    inline constexpr std::uint32_t kMoEOverlayNativeInvocationABIVersion = 1u;
    inline constexpr std::uint32_t kMoEOverlayNativeInvocationNoSlot = kMoEOverlayActivationOriginalNoSlot;

    /** Exact post-payload proof, written only by its original endpoint stream. */
    struct alignas(64) MoEOverlayNativeInvocationEndpointReceipt
    {
        MoEOverlayActivationEndpointStatus status{};
        /** Release publication, written only after every preceding payload operation. */
        std::uint64_t retired_generation = 0u;
    };

    /** Original immutable accepted admission plus its native endpoint receipts. */
    struct alignas(64) MoEOverlayNativeInvocationSlot
    {
        MoEOverlayActivationEpochIdentity identity{};
        std::uint64_t original_slot_index = 0u;
        std::uint32_t abi_version = kMoEOverlayNativeInvocationABIVersion;
        std::uint32_t reserved = 0u;
        /** Original host publication release word; zero means not issued. */
        std::uint64_t published_generation = 0u;
        /** Original immutable earliest watchdog observation from accepted admission. */
        std::uint64_t timeout_not_before_ns = 0u;
        /** Exception-only original cancellation publication; never a model-state shadow. */
        std::uint64_t cancelled_generation = 0u;
        MoEOverlayNativeInvocationEndpointReceipt continuation;
        MoEOverlayNativeInvocationEndpointReceipt follower;
    };

    /** Capture-stable physical view of the SAME original slot roster. */
    struct MoEOverlayNativeInvocationDeviceBinding
    {
        MoEOverlayActivationAdmissionPolicy policy =
            MoEOverlayActivationAdmissionPolicy::HostOrderedEpoch;
        MoEOverlayNativeInvocationSlot *original_slots = nullptr;
        std::uint32_t original_slot_count = 0u;

        /** @return Whether the declared owner has its exact required physical view. */
        [[nodiscard]] LLAMINAR_MOE_NATIVE_INVOCATION_HD constexpr bool valid() const noexcept
        {
            return policy == MoEOverlayActivationAdmissionPolicy::HostOrderedEpoch
                ? original_slots == nullptr && original_slot_count == 0u
                : policy == MoEOverlayActivationAdmissionPolicy::OriginalNativeSlots &&
                  original_slots != nullptr && original_slot_count != 0u;
        }
        /** @return Whether the original native owner must consume a checked slot. */
        [[nodiscard]] LLAMINAR_MOE_NATIVE_INVOCATION_HD constexpr bool deviceOwned() const noexcept
        { return policy == MoEOverlayActivationAdmissionPolicy::OriginalNativeSlots; }
        bool operator==(const MoEOverlayNativeInvocationDeviceBinding &) const = default;
    };

    /**
     * @brief Authenticate a published immutable original slot against its actual channel.
     * @param identity Snapshot acquired after this slot's original publication word.
     * @param published_generation Acquired release word from that same original slot.
     * @param channel Original immutable family/channel declaration.
     * @param physical_rows Exact captured row specialization.
     * @return True only for this original source, family, rows and generation.
     */
    [[nodiscard]] LLAMINAR_MOE_NATIVE_INVOCATION_HD LLAMINAR_MOE_NATIVE_INVOCATION_INLINE bool
    moeOverlayNativeInvocationAdmissionMatches(
        const MoEOverlayActivationEpochIdentity &identity,
        std::uint64_t published_generation,
        const MoEOverlayActivationChannelHeader &channel,
        std::int32_t physical_rows) noexcept
    {
        if (channel.magic!=kMoEOverlayActivationMagic ||
            channel.abi_version!=kMoEOverlayActivationABIVersion ||
            channel.buffer_count!=kMoEOverlayActivationBufferCount ||
            channel.channel_nonce==0u || channel.workspace_generation==0u ||
            channel.stage_manifest_digest==0u ||
            physical_rows <= 0 || published_generation == 0u ||
            published_generation != identity.epoch_generation ||
            identity.epoch_generation > kMoEOverlayActivationMaxGeneration ||
            identity.channel_nonce != channel.channel_nonce ||
            identity.workspace_generation != channel.workspace_generation ||
            identity.topology_fingerprint_low != channel.topology_fingerprint_low ||
            identity.topology_fingerprint_high != channel.topology_fingerprint_high ||
            identity.stage_manifest_digest != channel.stage_manifest_digest ||
            identity.source_world_rank != channel.source_world_rank ||
            identity.target_world_rank != channel.target_world_rank ||
            identity.source_participant_id != channel.source_participant_id ||
            identity.target_participant_id != channel.target_participant_id ||
            identity.source_tier_priority != channel.source_tier_priority ||
            identity.target_tier_priority != channel.target_tier_priority ||
            identity.source_domain_ordinal != channel.source_domain_ordinal ||
            identity.target_domain_ordinal != channel.target_domain_ordinal ||
            identity.lane_ordinal != channel.lane_ordinal ||
            identity.return_role != channel.return_role ||
            identity.cohort_target != channel.cohort_target ||
            identity.stage_count != channel.stage_count || identity.stage_count == 0u ||
            identity.graph_role >= 32u ||
            (channel.graph_role_mask & (std::uint32_t{1} << identity.graph_role)) == 0u ||
            identity.request_generation == 0u || identity.command_id == 0u ||
            identity.request_count == 0u || identity.logical_rows_per_request == 0u ||
            identity.logical_rows_per_request>identity.physical_rows_per_request ||
            identity.physical_rows_per_request == 0u ||
            static_cast<std::uint64_t>(identity.request_count) * identity.physical_rows_per_request !=
                static_cast<std::uint64_t>(physical_rows))
            return false;
        const auto digest = moeOverlayActivationIdentityDigest(identity);
        return identity.digest.valid() && identity.digest.low == digest.low && identity.digest.high == digest.high;
    }

    /**
     * @brief Check the exact original physical slot and acquired immutable publication snapshot.
     * @param slot Same original mapped record after acquiring published_generation.
     * @param index Original roster index from the capture-stable physical scan.
     * @param count Exact original roster count bound at cold admission.
     * @param channel Original immutable family/channel declaration.
     * @param physical_rows Exact captured row specialization.
     * @return True only for this current correctly versioned original slot.
     */
    [[nodiscard]] LLAMINAR_MOE_NATIVE_INVOCATION_HD inline bool
    moeOverlayNativeInvocationSlotMatches(
        const MoEOverlayNativeInvocationSlot &slot,std::uint32_t index,std::uint32_t count,
        const MoEOverlayActivationChannelHeader &channel,std::int32_t physical_rows) noexcept
    {
        return count!=0u && index<count && slot.original_slot_index==index && slot.identity.original_slot_index==index &&
            slot.abi_version==kMoEOverlayNativeInvocationABIVersion && slot.reserved==0u &&
            slot.timeout_not_before_ns!=0u && slot.cancelled_generation==0u &&
            moeOverlayNativeInvocationAdmissionMatches(slot.identity,slot.published_generation,channel,physical_rows);
    }

    /**
     * @brief Validate post-payload retirement without inferring it from semantic Complete.
     * @param receipt Snapshot after acquiring this endpoint's retired_generation.
     * @param identity Exact original immutable slot identity.
     * @return True only for a successful complete final-stage packet retirement.
     */
    [[nodiscard]] LLAMINAR_MOE_NATIVE_INVOCATION_HD inline bool
    moeOverlayNativeInvocationEndpointRetired(
        const MoEOverlayNativeInvocationEndpointReceipt &receipt,
        const MoEOverlayActivationEpochIdentity &identity) noexcept
    {
        const auto &status = receipt.status;
        const auto digest=moeOverlayActivationIdentityDigest(identity);
        return identity.digest.valid() && identity.digest.low==digest.low && identity.digest.high==digest.high &&
            identity.epoch_generation != 0u && identity.stage_count != 0u &&
            receipt.retired_generation == identity.epoch_generation &&
            status.payload_retired_generation == identity.epoch_generation &&
            status.generation == identity.epoch_generation &&
            status.digest.low == identity.digest.low && status.digest.high == identity.digest.high &&
            status.state == static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Complete) &&
            status.code == static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success) &&
            status.last_consumed_stage == static_cast<std::int32_t>(identity.stage_count - 1u) &&
            status.last_published_stage == static_cast<std::int32_t>(identity.stage_count - 1u);
    }

    static_assert(std::is_trivially_copyable_v<MoEOverlayNativeInvocationSlot>);
    static_assert(alignof(MoEOverlayNativeInvocationSlot) >= 64u);
}

#undef LLAMINAR_MOE_NATIVE_INVOCATION_HD

#undef LLAMINAR_MOE_NATIVE_INVOCATION_INLINE
