/**
 * @file MoEOverlayDeviceControllerDevice.inl
 * @brief Shared CUDA/HIP implementation of mapped ExpertOverlay control.
 *
 * This file is included by exactly one CUDA and one HIP launch bridge. Every
 * cross-device publication uses system-scope acquire/release ordering. The
 * continuation-root GPU is the only writer of global lifecycle and command
 * records; each group root writes only its own acknowledgement page.
 */

#pragma once

#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "execution/moe/DeviceMoERuntimeABI.h"
#include "execution/moe/MoEOverlayDeviceControllerKernels.h"
#include "kernels/common/DeviceMoEFloatingMatrixDesc.h"
#include "kernels/common/DeviceNativeVNNIMatrixDesc.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

namespace llaminar2::moe_overlay_controller_device
{
    /** Fixed block width used by the deterministic snapshot reduction. */
    inline constexpr std::uint32_t kControllerThreads = 256u;

    /** Per-block scratch for one deterministic topology-wide policy author. */
    struct DynamicPolicyScratch
    {
        /** One validated mapped snapshot plane per physical participant. */
        const std::uint64_t *participant_state_base[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint64_t expert_counts[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        /** Phase-pure demand in service-profile order: decode, prefill. */
        std::uint64_t phase_expert_counts[
            kMoEOverlayDeviceControllerDemandPhaseCount]
            [kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint64_t participant_load[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        /** Transaction-local device cache of immutable per-tier service cost. */
        std::uint64_t service_cost[
            kMoEOverlayDeviceControllerFabricMaxParticipants]
            [kMoEOverlayDeviceControllerDemandPhaseCount];
        /** Layer-local cache avoids repeated PCIe reads while scoring cycles. */
        std::uint64_t migration_transfer_and_repack_ns
            [kMoEOverlayDeviceControllerFabricMaxParticipants]
            [kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint64_t migration_inference_interference_ns
            [kMoEOverlayDeviceControllerFabricMaxParticipants]
            [kMoEOverlayDeviceControllerFabricMaxParticipants];
        /** Layer-local hysteresis generation for each expert. */
        std::uint64_t last_moved[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint64_t minimum_residency_generations;
        std::uint64_t payoff_horizon_tokens;
        std::uint64_t minimum_net_benefit_ns;
        std::int32_t current_owner[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        std::int32_t desired_owner[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        std::int32_t candidate_owner[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        std::int32_t participant_priority[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::int32_t participant_tier[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::int32_t priorities[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint32_t ordered_experts[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint32_t quotas[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint32_t remaining[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint32_t priority_participants[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint32_t cycle[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint32_t best_cycle[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint32_t dfs_source[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint32_t dfs_next_expert[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
        std::uint8_t excluded[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        /** Transaction-local mask used to enumerate disjoint cycle candidates. */
        std::uint8_t enumerated[
            kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint8_t visited[
            kMoEOverlayDeviceControllerFabricMaxParticipants];
    };

    /** Lexicographic service objective shared with the CPU policy oracle. */
    struct DynamicPlacementScore
    {
        std::uint64_t priority_cost = 0u;
        std::uint64_t same_priority_makespan = 0u;
    };

    /** Exact measured payoff and anti-oscillation verdict for one closed cycle. */
    struct DynamicCycleEconomyScore
    {
        std::uint64_t service_before_ns = 0u;
        std::uint64_t service_after_ns = 0u;
        std::uint64_t projected_service_gain_ns = 0u;
        std::uint64_t transfer_and_repack_ns = 0u;
        std::uint64_t inference_interference_ns = 0u;
        std::uint64_t projected_net_benefit_ns = 0u;
        bool residency_eligible = true;
        bool payoff_eligible = false;

        /** @return Whether both measured policy gates accept the cycle. */
        __device__ __forceinline__ bool eligible() const noexcept
        {
            return residency_eligible && payoff_eligible;
        }
    };

    /** Device-safe view of the host runtime's fixed expert descriptor ABI. */
    struct RuntimeExpertDescriptorView
    {
        DeviceNativeVNNIMatrixDesc gate;
        DeviceNativeVNNIMatrixDesc up;
        DeviceNativeVNNIMatrixDesc down;
        DeviceMoEFloatingMatrixDesc floating_gate;
        DeviceMoEFloatingMatrixDesc floating_up;
        DeviceMoEFloatingMatrixDesc floating_down;
        std::int32_t logical_expert_id;
        std::int32_t owner_participant;
        std::int32_t local_slot;
        std::uint32_t flags;
        DeviceMoEWeightFormat weight_format;
        std::uint32_t reserved;
    };

    /** Device-safe view of one complete double-buffered placement generation. */
    struct RuntimePlacementBankView
    {
        RuntimeExpertDescriptorView
            experts[kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint8_t
            local_compute_mask[kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint8_t
            replica_role[kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint32_t resident_participant_mask
            [kMoEOverlayDeviceControllerFabricMaxExperts];
        std::int32_t overlay_route_participant
            [kMoEOverlayDeviceControllerFabricMaxExperts];
        std::uint32_t epoch;
        std::uint32_t expert_count;
        std::uint32_t multi_resident_expert_count;
        std::uint32_t transient_placement_observed;
    };

    /** Prefix of one runtime layer; array stepping uses the reviewed ABI size. */
    struct RuntimeLayerPlacementPrefixView
    {
        std::uint32_t active_bank;
        std::uint32_t active_epoch;
        std::uint32_t expert_count;
        std::uint32_t top_k;
        RuntimePlacementBankView banks[kDeviceMoEOverlayEpochBankCount];
    };

    static_assert(sizeof(RuntimeExpertDescriptorView) == 240u);
    static_assert(
        sizeof(RuntimePlacementBankView) ==
        moe_runtime_abi::kPlacementBankBytes);
    static_assert(
        offsetof(RuntimePlacementBankView, overlay_route_participant) ==
        moe_runtime_abi::kOverlayRouteParticipantOffset);

    inline constexpr std::uint32_t kRuntimeExpertFlagValid = 1u << 0u;
    inline constexpr std::uint32_t kRuntimeExpertFlagResident = 1u << 1u;
    inline constexpr std::uint32_t kRuntimeExpertFlagPreferredOwner = 1u << 3u;
    inline constexpr std::uint32_t kRuntimeExpertFlagLocalCompute = 1u << 4u;
    inline constexpr std::uint32_t kRuntimeExpertFlagTransferSlot = 1u << 5u;
    inline constexpr std::uint8_t kRuntimeReplicaRoleNone = 0u;
    inline constexpr std::uint8_t kRuntimeReplicaRolePrimary = 1u;

    /** @return Raw fixed-width value for one protocol enum. */
    template <typename Enum>
    __device__ __forceinline__ std::uint32_t raw(Enum value) noexcept
    {
        return static_cast<std::uint32_t>(value);
    }

    /** @return A cache-volatile mapped-memory load. */
    template <typename Value>
    __device__ __forceinline__ Value loadPeerPublished(
        const Value *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        return __ldcv(address);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __builtin_nontemporal_load(address);
#else
        return *address;
#endif
    }

    /** @return One system-scope acquire load from mapped memory. */
    template <typename Value>
    __device__ __forceinline__ Value loadSystemAcquire(
        const Value *address) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<Value, cuda::thread_scope_system> reference(
            *const_cast<Value *>(address));
        return reference.load(cuda::memory_order_acquire);
#elif defined(__HIP_DEVICE_COMPILE__)
        return __hip_atomic_load(
            address, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        return *address;
#endif
    }

    /** @brief One system-scope release store into mapped memory. */
    template <typename Value>
    __device__ __forceinline__ void storeSystemRelease(
        Value *address,
        Value value) noexcept
    {
#if defined(__CUDA_ARCH__)
        cuda::atomic_ref<Value, cuda::thread_scope_system> reference(*address);
        reference.store(value, cuda::memory_order_release);
#elif defined(__HIP_DEVICE_COMPILE__)
        __hip_atomic_store(
            address, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
#else
        *address = value;
#endif
    }

    /** Yield one short interval while a peer device owns the next edge. */
    __device__ __forceinline__ void peerWaitBackoff() noexcept
    {
#if defined(__CUDA_ARCH__)
        __nanosleep(64u);
#elif defined(__HIP_DEVICE_COMPILE__)
        __builtin_amdgcn_s_sleep(1u);
#endif
    }

    /** @return A whole fixed-width record loaded without stale cache lines. */
    template <typename Record>
    __device__ __forceinline__ Record snapshotPeerRecord(
        const Record *record) noexcept
    {
        static_assert(sizeof(Record) % sizeof(std::uint64_t) == 0u);
        Record snapshot{};
        const auto *source = reinterpret_cast<const std::uint64_t *>(record);
        auto *destination = reinterpret_cast<std::uint64_t *>(&snapshot);
        for (std::size_t lane = 0u;
             lane < sizeof(Record) / sizeof(std::uint64_t);
             ++lane)
        {
            destination[lane] = loadPeerPublished(source + lane);
        }
        return snapshot;
    }

    /** @return Exact placement prefix for one layer in the host-reviewed ABI. */
    __device__ __forceinline__ RuntimeLayerPlacementPrefixView *runtimeLayer(
        DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer) noexcept
    {
        auto *base = reinterpret_cast<std::byte *>(runtime_layers);
        return reinterpret_cast<RuntimeLayerPlacementPrefixView *>(
            base + static_cast<std::uint64_t>(layer) *
                       moe_runtime_abi::kLayerRuntimeBytes);
    }

    /** @return Const exact placement prefix for one reviewed runtime layer. */
    __device__ __forceinline__ const RuntimeLayerPlacementPrefixView *
    runtimeLayer(
        const DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer) noexcept
    {
        const auto *base = reinterpret_cast<const std::byte *>(runtime_layers);
        return reinterpret_cast<const RuntimeLayerPlacementPrefixView *>(
            base + static_cast<std::uint64_t>(layer) *
                       moe_runtime_abi::kLayerRuntimeBytes);
    }

    /** @return Participant id stored in the non-placement tail of one layer. */
    __device__ __forceinline__ std::uint32_t runtimeParticipantId(
        const DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer) noexcept
    {
        const auto *base = reinterpret_cast<const std::byte *>(runtime_layers) +
            static_cast<std::uint64_t>(layer) *
                moe_runtime_abi::kLayerRuntimeBytes;
        return *reinterpret_cast<const std::uint32_t *>(
            base + moe_runtime_abi::kParticipantIdOffset);
    }

    /** @return Participant count stored in one layer's immutable geometry. */
    __device__ __forceinline__ std::uint32_t runtimeParticipantCount(
        const DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer) noexcept
    {
        const auto *base = reinterpret_cast<const std::byte *>(runtime_layers) +
            static_cast<std::uint64_t>(layer) *
                moe_runtime_abi::kLayerRuntimeBytes;
        return *reinterpret_cast<const std::uint32_t *>(
            base + moe_runtime_abi::kParticipantCountOffset);
    }

    /**
     * @return Whether the layer owns its embedded durable banks.
     *
     * MTP and request-local LLEP child tables point at the main table instead.
     * The topology controller must reject such a child binding because writing
     * its embedded scratch would not change the canonical inference placement.
     */
    __device__ __forceinline__ bool runtimeOwnsPlacementBanks(
        const DeviceMoELayerRuntime *runtime_layers,
        std::uint32_t layer) noexcept
    {
        const auto *base = reinterpret_cast<const std::byte *>(runtime_layers) +
            static_cast<std::uint64_t>(layer) *
                moe_runtime_abi::kLayerRuntimeBytes;
        return *reinterpret_cast<const void *const *>(
                   base + moe_runtime_abi::kOverlayPlacementBanksOffset) ==
               nullptr;
    }

    /** @return Device-safe completeness check for one NativeVNNI matrix. */
    __device__ __forceinline__ bool nativeMatrixReady(
        const DeviceNativeVNNIMatrixDesc &matrix) noexcept
    {
        return matrix.payload != nullptr && matrix.scales != nullptr &&
               matrix.n > 0 && matrix.k > 0 && matrix.blocks_per_row > 0u;
    }

    /** @return Device-safe completeness check for one floating matrix. */
    __device__ __forceinline__ bool floatingMatrixReady(
        const DeviceMoEFloatingMatrixDesc &matrix) noexcept
    {
        return matrix.data != nullptr && matrix.n > 0 && matrix.k > 0;
    }

    /** @return Whether one prepared arrival is a complete shape-consistent triple. */
    __device__ __forceinline__ bool runtimeArrivalReady(
        const RuntimeExpertDescriptorView &descriptor,
        std::uint32_t expert) noexcept
    {
        if (descriptor.logical_expert_id !=
                static_cast<std::int32_t>(expert) ||
            descriptor.local_slot < 0)
        {
            return false;
        }
        if (descriptor.weight_format == DeviceMoEWeightFormat::NativeVNNI)
        {
            return nativeMatrixReady(descriptor.gate) &&
                   nativeMatrixReady(descriptor.up) &&
                   nativeMatrixReady(descriptor.down) &&
                   descriptor.gate.n == descriptor.up.n &&
                   descriptor.gate.k == descriptor.up.k &&
                   descriptor.down.n == descriptor.gate.k &&
                   descriptor.down.k == descriptor.gate.n;
        }
        return deviceMoEWeightFormatIsFloating(descriptor.weight_format) &&
               floatingMatrixReady(descriptor.floating_gate) &&
               floatingMatrixReady(descriptor.floating_up) &&
               floatingMatrixReady(descriptor.floating_down) &&
               descriptor.floating_gate.n == descriptor.floating_up.n &&
               descriptor.floating_gate.k == descriptor.floating_up.k &&
               descriptor.floating_down.n == descriptor.floating_gate.k &&
               descriptor.floating_down.k == descriptor.floating_gate.n;
    }

    /** @return Device view of the descriptor prepared for one command ordinal. */
    __device__ __forceinline__ const RuntimeExpertDescriptorView &
    preparedArrival(
        const MoEOverlayDeviceRuntimePublicationBinding &binding,
        std::uint32_t ordinal) noexcept
    {
        return reinterpret_cast<const RuntimeExpertDescriptorView *>(
            binding.prepared_arrivals)[ordinal];
    }

    /** @return Group layout authenticated by one dense group id. */
    __device__ __forceinline__ const
    MoEOverlayDeviceControllerFabricGroupLayout *groupLayout(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint32_t group_id) noexcept
    {
        if (!binding.layout || !binding.groups ||
            group_id >= loadPeerPublished(&binding.layout->group_count))
        {
            return nullptr;
        }
        return binding.groups + group_id;
    }

    /** @return Mapped group record resolved only from an immutable offset. */
    __device__ __forceinline__ MoEOverlayDeviceControllerGroupRecord *groupRecord(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint32_t group_id) noexcept
    {
        const auto *layout = groupLayout(binding, group_id);
        if (!layout)
            return nullptr;
        const std::uint64_t offset =
            loadPeerPublished(&layout->group_record_offset);
        if (offset > binding.mapped_bytes ||
            binding.mapped_bytes - offset <
                sizeof(MoEOverlayDeviceControllerGroupRecord))
        {
            return nullptr;
        }
        return reinterpret_cast<MoEOverlayDeviceControllerGroupRecord *>(
            binding.mapped_base + offset);
    }

    /** @return One participant record resolved from immutable group topology. */
    __device__ __forceinline__
    MoEOverlayDeviceControllerParticipantRecord *participantRecord(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint32_t group_id,
        std::uint32_t member) noexcept
    {
        const auto *group = groupLayout(binding, group_id);
        if (!group || member >= group->participant_count ||
            group->participant_record_count != group->participant_count)
        {
            return nullptr;
        }
        const std::uint64_t offset = group->participant_records_offset;
        const std::uint64_t bytes =
            static_cast<std::uint64_t>(group->participant_record_count) *
            sizeof(MoEOverlayDeviceControllerParticipantRecord);
        if (offset > binding.mapped_bytes ||
            bytes > binding.mapped_bytes - offset)
        {
            return nullptr;
        }
        return reinterpret_cast<
                   MoEOverlayDeviceControllerParticipantRecord *>(
                   binding.mapped_base + offset) +
            member;
    }

    /**
     * @return Whether every topology participant has drained @p epoch locally.
     *
     * Readiness is a distinct phase from reclamation. A mapped follower may
     * acquire the continuation-selected retiring bank after its own device
     * first reports zero readers. The continuation reader remains live until
     * that follower returns, so topology-wide readiness cannot converge until
     * every such late acquisition has also released. Only this all-participant
     * release/acquire barrier makes an Empty transition safe on every device.
     */
    __device__ __forceinline__ bool allParticipantsRetirementReady(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint64_t epoch) noexcept
    {
        const std::uint32_t group_count =
            loadPeerPublished(&binding.layout->group_count);
        for (std::uint32_t group_id = 0u;
             group_id < group_count;
             ++group_id)
        {
            const auto *group = groupLayout(binding, group_id);
            if (!group)
                return false;
            for (std::uint32_t member = 0u;
                 member < group->participant_count;
                 ++member)
            {
                const auto *record =
                    participantRecord(binding, group_id, member);
                if (!record ||
                    loadPeerPublished(&record->magic) !=
                        kMoEOverlayDeviceControllerFabricMagic ||
                    loadPeerPublished(&record->version) !=
                        kMoEOverlayDeviceControllerFabricVersion ||
                    loadPeerPublished(&record->group_id) != group_id ||
                    loadPeerPublished(&record->participant_id) !=
                        loadPeerPublished(&group->participant_ids[member]) ||
                    loadPeerPublished(&record->topology_fingerprint) !=
                        binding.topology_fingerprint ||
                    loadSystemAcquire(&record->retirement_ready_epoch) != epoch)
                {
                    return false;
                }
            }
        }
        return true;
    }

    /** @return Whether immutable mapped topology matches this captured graph. */
    __device__ __forceinline__ bool validIdentity(
        const MoEOverlayDeviceControllerDeviceBinding &binding) noexcept
    {
        if (!binding.valid() ||
            loadPeerPublished(&binding.layout->magic) !=
                kMoEOverlayDeviceControllerFabricMagic ||
            loadPeerPublished(&binding.layout->version) !=
                kMoEOverlayDeviceControllerFabricVersion ||
            loadPeerPublished(&binding.layout->topology_fingerprint) !=
                binding.topology_fingerprint ||
            loadPeerPublished(&binding.layout->mapping_bytes) !=
                binding.mapped_bytes ||
            loadPeerPublished(
                &binding.layout->minimum_window_activations) == 0u ||
            loadPeerPublished(
                &binding.layout->maximum_cycles_per_wave) == 0u ||
            loadPeerPublished(
                &binding.layout->payload_geometry_fingerprint) == 0u ||
            binding.group_id >=
                loadPeerPublished(&binding.layout->group_count) ||
            binding.participant_id >=
                loadPeerPublished(&binding.layout->participant_count) ||
            loadPeerPublished(&binding.controller->magic) !=
                kMoEOverlayDeviceControllerMagic ||
            loadPeerPublished(&binding.controller->version) !=
                kMoEOverlayDeviceControllerVersion ||
            loadPeerPublished(&binding.controller->topology_fingerprint) !=
                binding.topology_fingerprint ||
            loadPeerPublished(&binding.local_group->magic) !=
                kMoEOverlayDeviceControllerMagic ||
            loadPeerPublished(&binding.local_group->group_id) !=
                binding.group_id ||
            loadPeerPublished(&binding.local_group->topology_fingerprint) !=
                binding.topology_fingerprint ||
            loadPeerPublished(&binding.local_transport->magic) !=
                kMoEOverlayDeviceControllerFabricMagic ||
            loadPeerPublished(&binding.local_transport->version) !=
                kMoEOverlayDeviceControllerFabricVersion ||
            loadPeerPublished(&binding.local_transport->group_id) !=
                binding.group_id ||
            loadPeerPublished(
                &binding.local_transport->topology_fingerprint) !=
                binding.topology_fingerprint ||
            loadPeerPublished(&binding.local_participant_record->magic) !=
                kMoEOverlayDeviceControllerFabricMagic ||
            loadPeerPublished(&binding.local_participant_record->version) !=
                kMoEOverlayDeviceControllerFabricVersion ||
            loadPeerPublished(
                &binding.local_participant_record->participant_id) !=
                binding.participant_id ||
            loadPeerPublished(&binding.local_participant_record->group_id) !=
                binding.group_id ||
            loadPeerPublished(
                &binding.local_participant_record->topology_fingerprint) !=
                binding.topology_fingerprint)
        {
            return false;
        }
        const auto metadata = snapshotPeerRecord(
            binding.participants + binding.participant_id);
        return metadata.participant_id == binding.participant_id &&
               metadata.group_id == binding.group_id &&
               metadata.flags == binding.role_flags;
    }

    /** @brief Leader-owned terminal error publication. */
    __device__ __forceinline__ void failLeader(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        MoEOverlayDeviceControllerError error,
        std::uint32_t group_id = 0xffffffffu) noexcept
    {
        if (!binding.controller || !binding.authorityLeader())
            return;
        if (loadSystemAcquire(&binding.controller->error_code) ==
            raw(MoEOverlayDeviceControllerError::None))
        {
            binding.controller->error_group_id = group_id;
            binding.controller->error_code = raw(error);
        }
        __threadfence_system();
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::Error));
    }

    /** @brief Group-root-owned fault publication observed by the leader. */
    __device__ __forceinline__ void failGroup(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        MoEOverlayDeviceControllerError error) noexcept
    {
        if (!binding.local_group || !binding.groupRoot())
            return;
        storeSystemRelease(&binding.local_group->status_code, raw(error));
    }

    /** @brief Participant-owned fault publication observed by its group root. */
    __device__ __forceinline__ void failParticipant(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        MoEOverlayDeviceControllerError error) noexcept
    {
        if (!binding.local_participant_record)
            return;
        storeSystemRelease(
            &binding.local_participant_record->status_code,
            raw(error));
    }

    /** @return Whether the global controller reached one exact live state. */
    __device__ __forceinline__ bool waitForState(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        MoEOverlayDeviceControllerState expected) noexcept
    {
        while (true)
        {
            const std::uint32_t state =
                loadSystemAcquire(&binding.controller->state);
            if (state == raw(MoEOverlayDeviceControllerState::Error))
                return false;
            if (state == raw(expected))
                return true;
            peerWaitBackoff();
        }
    }

    /**
     * @return Whether every group published @p expected into one record field.
     *
     * The leader polls only one cache line per controller group. A nonzero
     * group status is converted into the one global terminal error before the
     * kernel returns, releasing every peer wait loop.
     */
    enum class GroupWordField : std::uint32_t
    {
        SnapshotTransaction,
        PreparedTransaction,
        PublishedTransaction,
        RestoredTransaction,
        RetiredEpoch,
    };

    /** @return Selected monotonic group word with system acquire ordering. */
    __device__ __forceinline__ std::uint64_t loadGroupWord(
        const MoEOverlayDeviceControllerGroupRecord *record,
        GroupWordField field) noexcept
    {
        switch (field)
        {
        case GroupWordField::SnapshotTransaction:
            return loadSystemAcquire(&record->snapshot_transaction);
        case GroupWordField::PreparedTransaction:
            return loadSystemAcquire(&record->prepared_transaction);
        case GroupWordField::PublishedTransaction:
            return loadSystemAcquire(&record->published_transaction);
        case GroupWordField::RestoredTransaction:
            return loadSystemAcquire(&record->restored_transaction);
        case GroupWordField::RetiredEpoch:
            return loadSystemAcquire(&record->retired_epoch);
        }
        return 0u;
    }

    __device__ __forceinline__ bool waitForAllGroups(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        GroupWordField field,
        std::uint64_t expected) noexcept
    {
        const std::uint32_t count =
            loadPeerPublished(&binding.layout->group_count);
        while (true)
        {
            bool ready = true;
            for (std::uint32_t group = 0u; group < count; ++group)
            {
                auto *record = groupRecord(binding, group);
                if (!record)
                {
                    failLeader(
                        binding,
                        MoEOverlayDeviceControllerError::InvalidTopology,
                        group);
                    return false;
                }
                const std::uint32_t status =
                    loadPeerPublished(&record->status_code);
                if (status != raw(MoEOverlayDeviceControllerError::None))
                {
                    failLeader(
                        binding,
                        static_cast<MoEOverlayDeviceControllerError>(status),
                        group);
                    return false;
                }
                if (loadGroupWord(record, field) < expected)
                    ready = false;
            }
            if (ready)
                return true;
            if (loadSystemAcquire(&binding.controller->state) ==
                raw(MoEOverlayDeviceControllerState::Error))
            {
                return false;
            }
            peerWaitBackoff();
        }
    }

    /** Physical completion word selected from the host transport-only lane. */
    enum class TransportWordField : std::uint32_t
    {
        PreparedTransaction,
        PublishedTransaction,
        RetiredEpoch,
        RestoredTransaction,
    };

    /** @return One transport completion word with system acquire ordering. */
    __device__ __forceinline__ std::uint64_t loadTransportWord(
        const MoEOverlayDeviceControllerTransportRecord *record,
        TransportWordField field) noexcept
    {
        switch (field)
        {
        case TransportWordField::PreparedTransaction:
            return loadSystemAcquire(&record->prepared_transaction);
        case TransportWordField::PublishedTransaction:
            return loadSystemAcquire(&record->published_transaction);
        case TransportWordField::RetiredEpoch:
            return loadSystemAcquire(&record->retired_epoch);
        case TransportWordField::RestoredTransaction:
            return loadSystemAcquire(&record->restored_transaction);
        }
        return 0u;
    }

    /**
     * @return Whether physical transport authenticated this command and phase.
     *
     * The host owns only this record. The group-root GPU checks both immutable
     * transaction identity and the independently re-derived command digest
     * before converting transport readiness into a controller acknowledgement.
     */
    __device__ __forceinline__ bool waitForLocalTransport(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        TransportWordField field,
        std::uint64_t transaction,
        std::uint64_t expected_word,
        std::uint64_t command_digest) noexcept
    {
        while (true)
        {
            const std::uint32_t status =
                loadSystemAcquire(&binding.local_transport->status_code);
            if (status != raw(MoEOverlayDeviceControllerError::None))
            {
                failGroup(
                    binding,
                    status <= raw(
                                  MoEOverlayDeviceControllerError::
                                      PhysicalTransportFailure)
                        ? static_cast<MoEOverlayDeviceControllerError>(status)
                        : MoEOverlayDeviceControllerError::
                              PhysicalTransportFailure);
                return false;
            }
            const std::uint64_t observed_transaction = loadSystemAcquire(
                &binding.local_transport->command_transaction);
            if (observed_transaction > transaction)
            {
                failGroup(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidTransaction);
                return false;
            }
            if (observed_transaction == transaction &&
                loadPeerPublished(
                    &binding.local_transport->command_digest) ==
                    command_digest &&
                loadTransportWord(binding.local_transport, field) ==
                    expected_word)
            {
                return true;
            }
            if (loadSystemAcquire(&binding.controller->state) ==
                raw(MoEOverlayDeviceControllerState::Error))
            {
                return false;
            }
            peerWaitBackoff();
        }
    }

    /** Participant-owned lifecycle word selected by one group-root wait. */
    enum class ParticipantWordField : std::uint32_t
    {
        SnapshotTransaction,
        PreparedTransaction,
        PublishedTransaction,
        RetiredEpoch,
        RestoredTransaction,
    };

    /** @return Selected participant word with system acquire ordering. */
    __device__ __forceinline__ std::uint64_t loadParticipantWord(
        const MoEOverlayDeviceControllerParticipantRecord *record,
        ParticipantWordField field) noexcept
    {
        switch (field)
        {
        case ParticipantWordField::SnapshotTransaction:
            return loadSystemAcquire(&record->snapshot_transaction);
        case ParticipantWordField::PreparedTransaction:
            return loadSystemAcquire(&record->prepared_transaction);
        case ParticipantWordField::PublishedTransaction:
            return loadSystemAcquire(&record->published_transaction);
        case ParticipantWordField::RetiredEpoch:
            return loadSystemAcquire(&record->retired_epoch);
        case ParticipantWordField::RestoredTransaction:
            return loadSystemAcquire(&record->restored_transaction);
        }
        return 0u;
    }

    /** @return Whether every member release-published one lifecycle edge. */
    __device__ __forceinline__ bool waitForAllGroupParticipants(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        ParticipantWordField field,
        std::uint64_t expected) noexcept
    {
        const auto *group = groupLayout(binding, binding.group_id);
        if (!group || !binding.group_participant_records ||
            group->participant_record_count != group->participant_count)
        {
            failGroup(binding, MoEOverlayDeviceControllerError::InvalidGroup);
            return false;
        }
        while (true)
        {
            bool ready = true;
            for (std::uint32_t member = 0u;
                 member < group->participant_count;
                 ++member)
            {
                const auto *record =
                    binding.group_participant_records + member;
                const std::uint32_t expected_participant =
                    loadPeerPublished(&group->participant_ids[member]);
                if (loadPeerPublished(&record->magic) !=
                        kMoEOverlayDeviceControllerFabricMagic ||
                    loadPeerPublished(&record->version) !=
                        kMoEOverlayDeviceControllerFabricVersion ||
                    loadPeerPublished(&record->participant_id) !=
                        expected_participant ||
                    loadPeerPublished(&record->group_id) != binding.group_id ||
                    loadPeerPublished(&record->topology_fingerprint) !=
                        binding.topology_fingerprint)
                {
                    failGroup(
                        binding,
                        MoEOverlayDeviceControllerError::InvalidTopology);
                    return false;
                }
                const std::uint32_t status =
                    loadPeerPublished(&record->status_code);
                if (status != raw(MoEOverlayDeviceControllerError::None))
                {
                    failGroup(
                        binding,
                        static_cast<MoEOverlayDeviceControllerError>(status));
                    return false;
                }
                if (loadParticipantWord(record, field) < expected)
                {
                    ready = false;
                }
            }
            if (ready)
                return true;
            if (loadSystemAcquire(&binding.controller->state) ==
                raw(MoEOverlayDeviceControllerState::Error))
            {
                return false;
            }
            peerWaitBackoff();
        }
    }

    /** @return Saturating sum used for snapshot observation evidence. */
    __device__ __forceinline__ std::uint64_t saturatingAdd(
        std::uint64_t left,
        std::uint64_t right) noexcept
    {
        const std::uint64_t sum = left + right;
        return sum < left ? 0xffffffffffffffffULL : sum;
    }

    /** @return Stable indexed contribution to a commutative snapshot digest. */
    __device__ __forceinline__ std::uint64_t digestWord(
        std::uint64_t word,
        std::uint64_t index) noexcept
    {
        std::uint64_t value =
            word ^ (index + 0x9e3779b97f4a7c15ULL);
        value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31u);
    }

    /** @return Saturating product used by the priority-rank objective. */
    __device__ __forceinline__ std::uint64_t saturatingMultiply(
        std::uint64_t left,
        std::uint64_t right) noexcept
    {
        if (left == 0u || right == 0u)
            return 0u;
        return left > 0xffffffffffffffffULL / right
                   ? 0xffffffffffffffffULL
                   : left * right;
    }

    /**
     * @return One participant's collected word resolved through immutable group
     *         membership, or false when the topology cannot address it.
     */
    __device__ __forceinline__ bool participantStateWord(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint32_t participant,
        std::uint32_t layer,
        std::uint32_t expert,
        std::uint64_t *word) noexcept
    {
        if (!word || participant >= binding.layout->participant_count ||
            layer >= binding.layout->num_layers ||
            expert >= binding.layout->num_experts)
        {
            return false;
        }
        const auto metadata = snapshotPeerRecord(
            binding.participants + participant);
        const auto *group = groupLayout(binding, metadata.group_id);
        if (!group || metadata.participant_id != participant)
            return false;

        std::uint32_t member_index = group->participant_count;
        for (std::uint32_t member = 0u;
             member < group->participant_count;
             ++member)
        {
            if (loadPeerPublished(&group->participant_ids[member]) ==
                participant)
            {
                member_index = member;
                break;
            }
        }
        const std::uint64_t words_per_participant =
            static_cast<std::uint64_t>(binding.layout->num_layers) *
            binding.layout->num_experts;
        if (member_index == group->participant_count ||
            group->collected_state_words !=
                words_per_participant * group->participant_count)
        {
            return false;
        }
        const std::uint64_t index =
            static_cast<std::uint64_t>(member_index) *
                words_per_participant +
            static_cast<std::uint64_t>(layer) *
                binding.layout->num_experts +
            expert;
        const std::uint64_t byte_offset =
            group->collected_state_offset +
            index * sizeof(std::uint64_t);
        if (byte_offset > binding.mapped_bytes ||
            binding.mapped_bytes - byte_offset < sizeof(std::uint64_t))
        {
            return false;
        }
        *word = loadPeerPublished(
            reinterpret_cast<const std::uint64_t *>(
                binding.mapped_base + byte_offset));
        return true;
    }

    /**
     * @return Validated beginning of one participant's contiguous snapshot.
     *
     * Policy used to repeat topology discovery for every `(layer, expert)`
     * word. On a heterogeneous mapped fabric that multiplied one useful
     * cache-volatile load into many PCIe reads. Resolve immutable group
     * membership once per transaction; the hot loop then reads exactly one
     * published activation word per participant.
     */
    __device__ __forceinline__ const std::uint64_t *
    participantStateBase(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint32_t participant,
        const MoEOverlayDeviceControllerParticipantMetadata &metadata) noexcept
    {
        if (participant >= binding.layout->participant_count ||
            metadata.participant_id != participant)
        {
            return nullptr;
        }
        const auto *group = groupLayout(binding, metadata.group_id);
        if (!group)
            return nullptr;

        std::uint32_t member_index = group->participant_count;
        for (std::uint32_t member = 0u;
             member < group->participant_count;
             ++member)
        {
            if (loadPeerPublished(&group->participant_ids[member]) ==
                participant)
            {
                member_index = member;
                break;
            }
        }
        const std::uint64_t words_per_participant =
            static_cast<std::uint64_t>(binding.layout->num_layers) *
            binding.layout->num_experts;
        if (member_index == group->participant_count ||
            group->collected_state_words !=
                words_per_participant * group->participant_count)
        {
            return nullptr;
        }
        const std::uint64_t byte_offset =
            group->collected_state_offset +
            static_cast<std::uint64_t>(member_index) *
                words_per_participant * sizeof(std::uint64_t);
        if (byte_offset > binding.mapped_bytes ||
            binding.mapped_bytes - byte_offset <
                words_per_participant * sizeof(std::uint64_t))
        {
            return nullptr;
        }
        return reinterpret_cast<const std::uint64_t *>(
            binding.mapped_base + byte_offset);
    }

    /** @return Dense rank of one opaque integer priority. */
    __device__ __forceinline__ std::uint32_t dynamicPriorityRank(
        const DynamicPolicyScratch &scratch,
        std::uint32_t priority_count,
        std::int32_t priority) noexcept
    {
        for (std::uint32_t rank = 0u; rank < priority_count; ++rank)
        {
            if (scratch.priorities[rank] == priority)
                return rank;
        }
        return priority_count;
    }

    /** Score one complete layer assignment using fixed arithmetic order. */
    __device__ __forceinline__ DynamicPlacementScore scoreDynamicLayer(
        DynamicPolicyScratch &scratch,
        const std::int32_t *owners,
        std::uint32_t participant_count,
        std::uint32_t expert_count,
        std::uint32_t priority_count) noexcept
    {
        DynamicPlacementScore score{};
        for (std::uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            scratch.participant_load[participant] = 0u;
        }
        for (std::uint32_t expert = 0u; expert < expert_count; ++expert)
        {
            const auto owner = static_cast<std::uint32_t>(owners[expert]);
            const std::uint64_t count = scratch.expert_counts[expert];
            scratch.participant_load[owner] = saturatingAdd(
                scratch.participant_load[owner], count);
            score.priority_cost = saturatingAdd(
                score.priority_cost,
                saturatingMultiply(
                    count,
                    dynamicPriorityRank(
                        scratch,
                        priority_count,
                        scratch.participant_priority[owner])));
        }
        for (std::uint32_t priority = 0u;
             priority < priority_count;
             ++priority)
        {
            std::uint64_t maximum = 0u;
            for (std::uint32_t participant = 0u;
                 participant < participant_count;
                 ++participant)
            {
                if (scratch.participant_priority[participant] ==
                    scratch.priorities[priority])
                {
                    const std::uint64_t load =
                        scratch.participant_load[participant];
                    maximum = load > maximum ? load : maximum;
                }
            }
            score.same_priority_makespan = saturatingAdd(
                score.same_priority_makespan, maximum);
        }
        return score;
    }

    /** @return Whether the selected cycle stays in one materially skewed priority. */
    __device__ __forceinline__ bool samePriorityDynamicCycleIsEligible(
        DynamicPolicyScratch &scratch,
        std::uint32_t cycle_length,
        std::uint32_t participant_count,
        std::uint32_t expert_count,
        std::uint32_t imbalance_threshold_per_mille) noexcept
    {
        if (cycle_length == 0u)
            return false;

        std::int32_t cycle_priority = 0;
        bool selected_priority = false;
        for (std::uint32_t edge = 0u; edge < cycle_length; ++edge)
        {
            const std::uint32_t expert = scratch.cycle[edge];
            const auto source = static_cast<std::uint32_t>(
                scratch.current_owner[expert]);
            const auto destination = static_cast<std::uint32_t>(
                scratch.desired_owner[expert]);
            const std::int32_t source_priority =
                scratch.participant_priority[source];
            const std::int32_t destination_priority =
                scratch.participant_priority[destination];
            if (source_priority != destination_priority)
                return false;
            if (!selected_priority)
            {
                cycle_priority = source_priority;
                selected_priority = true;
            }
            else if (cycle_priority != source_priority)
            {
                return false;
            }
        }

        for (std::uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            scratch.participant_load[participant] = 0u;
        }
        for (std::uint32_t expert = 0u; expert < expert_count; ++expert)
        {
            const auto owner = static_cast<std::uint32_t>(
                scratch.current_owner[expert]);
            scratch.participant_load[owner] = saturatingAdd(
                scratch.participant_load[owner],
                scratch.expert_counts[expert]);
        }
        return moe_rebalance_policy::samePriorityLoadIsImbalanced(
            scratch.participant_load,
            scratch.participant_priority,
            participant_count,
            cycle_priority,
            imbalance_threshold_per_mille);
    }

    /** @return Immutable tier/layer/phase service cost in profile order. */
    __device__ __forceinline__ std::uint64_t dynamicEconomyServiceCost(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint32_t tier,
        std::uint32_t layer,
        std::uint32_t phase) noexcept
    {
        const std::uint64_t offset =
            (static_cast<std::uint64_t>(tier) * binding.layout->num_layers +
             layer) *
                kMoEOverlayDeviceControllerEconomyServicePhaseCount +
            phase;
        return loadPeerPublished(binding.economy_service_costs + offset);
    }

    /** @return Directed measured movement price for one complete expert. */
    __device__ __forceinline__ MoEOverlayDeviceControllerMigrationCost
    dynamicEconomyMigrationCost(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint32_t source,
        std::uint32_t destination,
        std::uint32_t layer) noexcept
    {
        const std::uint64_t offset =
            (static_cast<std::uint64_t>(source) *
                 binding.layout->participant_count +
             destination) *
                binding.layout->num_layers +
            layer;
        return binding.economy_migration_costs[offset];
    }

    /**
     * @return Conservative floor of `gain * horizon * top_k / activations`.
     *
     * Histories contain routed activations rather than tokens. The immutable
     * router fan-out converts those units without host-maintained token state.
     * Splitting quotient and remainder avoids overflowing ordinary histories;
     * a genuinely unrepresentable result saturates and remains conservative
     * for the strict positive-payoff decision.
     */
    __device__ __forceinline__ std::uint64_t projectDynamicServiceGain(
        std::uint64_t gain,
        std::uint64_t payoff_horizon_tokens,
        std::uint32_t routed_experts_per_token,
        std::uint64_t observed_activations) noexcept
    {
        if (gain == 0u || payoff_horizon_tokens == 0u ||
            routed_experts_per_token == 0u || observed_activations == 0u)
        {
            return 0u;
        }
        const std::uint64_t scale = saturatingMultiply(
            payoff_horizon_tokens, routed_experts_per_token);
        const std::uint64_t whole = saturatingMultiply(
            gain / observed_activations, scale);
        const std::uint64_t remainder = saturatingMultiply(
            gain % observed_activations, scale) /
            observed_activations;
        return saturatingAdd(whole, remainder);
    }

    /**
     * @brief Price one complete capacity-preserving cycle from measured evidence.
     *
     * Cross-tier work uses exact phase-weighted service deltas. A pure same-tier
     * cycle cannot change aggregate tier work, so it instead prices the change
     * in the tier's parallel participant makespan. Reciprocal pair calibration
     * already measured both directions together and is charged once at the
     * slower edge; longer cycles remain conservatively additive.
     */
    __device__ __forceinline__ DynamicCycleEconomyScore
    scoreDynamicCycleEconomy(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        DynamicPolicyScratch &scratch,
        std::uint32_t cycle_length,
        std::uint32_t layer,
        std::uint32_t participant_count,
        std::uint32_t expert_count,
        std::uint64_t transaction,
        std::uint32_t minimum_improvement_per_mille) noexcept
    {
        DynamicCycleEconomyScore score{};
        if (cycle_length == 0u)
            return score;

        const auto first_source = static_cast<std::uint32_t>(
            scratch.current_owner[scratch.cycle[0]]);
        const std::int32_t selected_tier =
            scratch.participant_tier[first_source];
        bool pure_same_tier = selected_tier >= 0;
        for (std::uint32_t edge = 0u; edge < cycle_length; ++edge)
        {
            const std::uint32_t expert = scratch.cycle[edge];
            const auto source = static_cast<std::uint32_t>(
                scratch.current_owner[expert]);
            const auto destination = static_cast<std::uint32_t>(
                scratch.desired_owner[expert]);
            pure_same_tier = pure_same_tier &&
                scratch.participant_tier[source] == selected_tier &&
                scratch.participant_tier[destination] == selected_tier;
        }

        constexpr std::uint32_t kMeasuredDemandPhases =
            kMoEOverlayDeviceControllerDemandPhaseCount;
        if (pure_same_tier)
        {
            for (std::uint32_t phase = 0u;
                 phase < kMeasuredDemandPhases;
                 ++phase)
            {
                for (std::uint32_t participant = 0u;
                     participant < participant_count;
                     ++participant)
                {
                    scratch.participant_load[participant] = 0u;
                }
                for (std::uint32_t expert = 0u;
                     expert < expert_count;
                     ++expert)
                {
                    const auto owner = static_cast<std::uint32_t>(
                        scratch.current_owner[expert]);
                    if (scratch.participant_tier[owner] == selected_tier)
                    {
                        scratch.participant_load[owner] = saturatingAdd(
                            scratch.participant_load[owner],
                            scratch.phase_expert_counts[phase][expert]);
                    }
                }
                std::uint64_t before_maximum = 0u;
                for (std::uint32_t participant = 0u;
                     participant < participant_count;
                     ++participant)
                {
                    if (scratch.participant_tier[participant] == selected_tier &&
                        scratch.participant_load[participant] > before_maximum)
                    {
                        before_maximum = scratch.participant_load[participant];
                    }
                    scratch.participant_load[participant] = 0u;
                }
                for (std::uint32_t expert = 0u;
                     expert < expert_count;
                     ++expert)
                {
                    const auto owner = static_cast<std::uint32_t>(
                        scratch.candidate_owner[expert]);
                    if (scratch.participant_tier[owner] == selected_tier)
                    {
                        scratch.participant_load[owner] = saturatingAdd(
                            scratch.participant_load[owner],
                            scratch.phase_expert_counts[phase][expert]);
                    }
                }
                std::uint64_t after_maximum = 0u;
                for (std::uint32_t participant = 0u;
                     participant < participant_count;
                     ++participant)
                {
                    if (scratch.participant_tier[participant] == selected_tier &&
                        scratch.participant_load[participant] > after_maximum)
                    {
                        after_maximum = scratch.participant_load[participant];
                    }
                }
                const std::uint64_t service = scratch.service_cost
                    [static_cast<std::uint32_t>(selected_tier)][phase];
                score.service_before_ns = saturatingAdd(
                    score.service_before_ns,
                    saturatingMultiply(before_maximum, service));
                score.service_after_ns = saturatingAdd(
                    score.service_after_ns,
                    saturatingMultiply(after_maximum, service));
            }
        }
        else
        {
            for (std::uint32_t edge = 0u; edge < cycle_length; ++edge)
            {
                const std::uint32_t expert = scratch.cycle[edge];
                const auto source = static_cast<std::uint32_t>(
                    scratch.current_owner[expert]);
                const auto destination = static_cast<std::uint32_t>(
                    scratch.desired_owner[expert]);
                const auto source_tier = static_cast<std::uint32_t>(
                    scratch.participant_tier[source]);
                const auto destination_tier = static_cast<std::uint32_t>(
                    scratch.participant_tier[destination]);
                for (std::uint32_t phase = 0u;
                     phase < kMeasuredDemandPhases;
                     ++phase)
                {
                    const std::uint64_t demand =
                        scratch.phase_expert_counts[phase][expert];
                    score.service_before_ns = saturatingAdd(
                        score.service_before_ns,
                        saturatingMultiply(
                            demand,
                            scratch.service_cost[source_tier][phase]));
                    score.service_after_ns = saturatingAdd(
                        score.service_after_ns,
                        saturatingMultiply(
                            demand,
                            scratch.service_cost[destination_tier][phase]));
                }
            }
        }

        const bool reciprocal_pair = cycle_length == 2u &&
            scratch.current_owner[scratch.cycle[0]] ==
                scratch.desired_owner[scratch.cycle[1]] &&
            scratch.current_owner[scratch.cycle[1]] ==
                scratch.desired_owner[scratch.cycle[0]];
        for (std::uint32_t edge = 0u; edge < cycle_length; ++edge)
        {
            const std::uint32_t expert = scratch.cycle[edge];
            const auto source = static_cast<std::uint32_t>(
                scratch.current_owner[expert]);
            const auto destination = static_cast<std::uint32_t>(
                scratch.desired_owner[expert]);
            const std::uint64_t movement_transfer_and_repack_ns =
                scratch.migration_transfer_and_repack_ns
                    [source][destination];
            const std::uint64_t movement_inference_interference_ns =
                scratch.migration_inference_interference_ns
                    [source][destination];
            if (reciprocal_pair)
            {
                score.transfer_and_repack_ns =
                    movement_transfer_and_repack_ns >
                            score.transfer_and_repack_ns
                        ? movement_transfer_and_repack_ns
                        : score.transfer_and_repack_ns;
                score.inference_interference_ns =
                    movement_inference_interference_ns >
                            score.inference_interference_ns
                        ? movement_inference_interference_ns
                        : score.inference_interference_ns;
            }
            else
            {
                score.transfer_and_repack_ns = saturatingAdd(
                    score.transfer_and_repack_ns,
                    movement_transfer_and_repack_ns);
                score.inference_interference_ns = saturatingAdd(
                    score.inference_interference_ns,
                    movement_inference_interference_ns);
            }

            const std::uint64_t last_moved = scratch.last_moved[expert];
            const std::uint64_t minimum_residency =
                scratch.minimum_residency_generations;
            if (last_moved !=
                    kMoEOverlayDeviceControllerNeverMovedGeneration &&
                (transaction < last_moved ||
                 transaction - last_moved < minimum_residency))
            {
                score.residency_eligible = false;
            }
        }

        if (!moe_rebalance_policy::relativeReductionAtLeastPerMille(
                score.service_before_ns,
                score.service_after_ns,
                minimum_improvement_per_mille))
        {
            return score;
        }
        const std::uint64_t gain =
            score.service_before_ns - score.service_after_ns;
        std::uint64_t observed_activations = 0u;
        for (std::uint32_t expert = 0u; expert < expert_count; ++expert)
        {
            observed_activations = saturatingAdd(
                observed_activations, scratch.expert_counts[expert]);
        }
        score.projected_service_gain_ns = projectDynamicServiceGain(
            gain,
            scratch.payoff_horizon_tokens,
            binding.layout->routed_experts_per_token,
            observed_activations);
        const std::uint64_t measured_cost = saturatingAdd(
            score.transfer_and_repack_ns,
            score.inference_interference_ns);
        if (score.projected_service_gain_ns > measured_cost)
        {
            score.projected_net_benefit_ns =
                score.projected_service_gain_ns - measured_cost;
        }
        score.payoff_eligible = score.projected_net_benefit_ns >
            scratch.minimum_net_benefit_ns;
        return score;
    }

    /** @return Whether left outranks right under deterministic measured payoff. */
    __device__ __forceinline__ bool dynamicEconomyScoreBetter(
        const DynamicCycleEconomyScore &left,
        const DynamicCycleEconomyScore &right) noexcept
    {
        return left.projected_net_benefit_ns !=
                       right.projected_net_benefit_ns
                   ? left.projected_net_benefit_ns >
                         right.projected_net_benefit_ns
                   : left.projected_service_gain_ns >
                         right.projected_service_gain_ns;
    }

    /**
     * Find the same canonical simple cycle as the CPU differential oracle.
     * The balanced owner/target graph guarantees every selected edge belongs
     * to a cycle; the bounded iterative DFS avoids device recursion.
     */
    __device__ __forceinline__ std::uint32_t findDynamicCycle(
        DynamicPolicyScratch &scratch,
        const std::uint8_t *excluded,
        std::uint32_t participant_count,
        std::uint32_t expert_count) noexcept
    {
        for (std::uint32_t first = 0u; first < expert_count; ++first)
        {
            if (excluded[first] != 0u ||
                scratch.current_owner[first] ==
                    scratch.desired_owner[first])
            {
                continue;
            }
            const auto start = static_cast<std::uint32_t>(
                scratch.current_owner[first]);
            const auto next = static_cast<std::uint32_t>(
                scratch.desired_owner[first]);
            scratch.cycle[0] = first;
            if (next == start)
                return 1u;
            for (std::uint32_t participant = 0u;
                 participant < participant_count;
                 ++participant)
            {
                scratch.visited[participant] = 0u;
            }
            scratch.visited[start] = 1u;
            scratch.visited[next] = 1u;
            std::uint32_t depth = 1u;
            scratch.dfs_source[depth] = next;
            scratch.dfs_next_expert[depth] = 0u;

            while (true)
            {
                bool descended = false;
                const std::uint32_t source = scratch.dfs_source[depth];
                for (std::uint32_t expert =
                         scratch.dfs_next_expert[depth];
                     expert < expert_count;
                     ++expert)
                {
                    scratch.dfs_next_expert[depth] = expert + 1u;
                    if (excluded[expert] != 0u || expert == first ||
                        scratch.current_owner[expert] !=
                            static_cast<std::int32_t>(source) ||
                        scratch.current_owner[expert] ==
                            scratch.desired_owner[expert])
                    {
                        continue;
                    }
                    const auto destination = static_cast<std::uint32_t>(
                        scratch.desired_owner[expert]);
                    scratch.cycle[depth] = expert;
                    if (destination == start)
                        return depth + 1u;
                    if (scratch.visited[destination] == 0u &&
                        depth + 1u < participant_count)
                    {
                        scratch.visited[destination] = 1u;
                        ++depth;
                        scratch.dfs_source[depth] = destination;
                        scratch.dfs_next_expert[depth] = 0u;
                        descended = true;
                        break;
                    }
                }
                if (descended)
                    continue;
                if (depth == 1u)
                    break;
                scratch.visited[scratch.dfs_source[depth]] = 0u;
                --depth;
            }
        }
        return 0u;
    }

    /** Begin a new policy transaction on the sole leader. */
    __device__ __forceinline__ void beginTransaction(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.authorityLeader())
        {
            failGroup(binding, MoEOverlayDeviceControllerError::InvalidControl);
            return;
        }
        const std::uint32_t state =
            loadSystemAcquire(&binding.controller->state);
        if (state != raw(MoEOverlayDeviceControllerState::Idle) &&
            state != raw(MoEOverlayDeviceControllerState::Complete))
        {
            failLeader(binding, MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        const std::uint64_t previous =
            loadSystemAcquire(&binding.controller->transaction_id);
        const std::uint64_t base =
            loadSystemAcquire(&binding.controller->current_durable_epoch);
        if (previous == 0xffffffffffffffffULL || base == 0u ||
            (launch.transaction_kind ==
                 MoEOverlayDeviceControllerTransactionKind::DynamicPlacement &&
             base == 0xffffffffffffffffULL))
        {
            failLeader(binding, MoEOverlayDeviceControllerError::EpochOverflow);
            return;
        }
        const std::uint64_t transaction = previous + 1u;
        const std::uint64_t candidate =
            launch.transaction_kind ==
                    MoEOverlayDeviceControllerTransactionKind::DynamicPlacement
                ? base + 1u
                : base;

        *binding.command = MoEOverlayDeviceControllerCommandHeader{};
        binding.command->topology_fingerprint = binding.topology_fingerprint;
        binding.controller->transaction_kind = raw(launch.transaction_kind);
        binding.command->demand_phase = raw(launch.demand_phase);
        binding.controller->base_epoch = base;
        binding.controller->candidate_epoch = candidate;
        binding.controller->command_transaction = 0u;
        binding.controller->commit_transaction = 0u;
        binding.controller->admission_transaction = 0u;
        binding.controller->active_llep_transaction = 0u;
        binding.controller->error_code =
            raw(MoEOverlayDeviceControllerError::None);
        binding.controller->error_group_id = 0xffffffffu;
        __threadfence_system();
        storeSystemRelease(
            &binding.controller->transaction_id, transaction);
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::CollectingSnapshots));
    }

    /** Deterministically reduce and publish one group-owned snapshot page. */
    __device__ __forceinline__ void publishParticipantSnapshot(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        __shared__ std::uint64_t digests[kControllerThreads];
        __shared__ std::uint64_t observations[kControllerThreads];
        __shared__ std::uint32_t ready;
        const auto &binding = launch.binding;
        if (threadIdx.x == 0u)
        {
            ready = waitForState(
                        binding,
                        MoEOverlayDeviceControllerState::CollectingSnapshots)
                        ? 1u
                        : 0u;
            if (ready == 0u)
            {
                failParticipant(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
            }
        }
        __syncthreads();
        if (ready == 0u)
            return;

        const std::uint64_t words =
            static_cast<std::uint64_t>(binding.layout->num_layers) *
            binding.layout->num_experts;
        std::uint64_t digest = 0u;
        std::uint64_t count = 0u;
        for (std::uint64_t index = threadIdx.x;
             index < words;
             index += blockDim.x)
        {
            const std::uint64_t word = loadPeerPublished(
                binding.participant_collected_state + index);
            digest ^= digestWord(word, index);
            count = saturatingAdd(
                count,
                moe_rebalance_policy::collectedStateActivationCount(word));
        }
        digests[threadIdx.x] = digest;
        observations[threadIdx.x] = count;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2u;
             stride != 0u;
             stride >>= 1u)
        {
            if (threadIdx.x < stride)
            {
                digests[threadIdx.x] ^= digests[threadIdx.x + stride];
                observations[threadIdx.x] = saturatingAdd(
                    observations[threadIdx.x],
                    observations[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x != 0u)
            return;

        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        const std::uint64_t prior = loadSystemAcquire(
            &binding.local_participant_record->snapshot_transaction);
        if (transaction == 0u || prior > transaction)
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidTransaction);
            return;
        }
        binding.local_participant_record->snapshot_digest =
            digests[0] == 0u ? 1u : digests[0];
        binding.local_participant_record->snapshot_observations =
            observations[0];
        binding.local_participant_record->status_code =
            raw(MoEOverlayDeviceControllerError::None);
        __threadfence_system();
        storeSystemRelease(
            &binding.local_participant_record->snapshot_transaction,
            transaction);
    }

    /** Deterministically reduce and publish one group-owned snapshot page. */
    __device__ __forceinline__ void publishGroupSnapshot(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        __shared__ std::uint64_t digests[kControllerThreads];
        __shared__ std::uint64_t observations[kControllerThreads];
        __shared__ std::uint32_t ready;
        const auto &binding = launch.binding;
        if (threadIdx.x == 0u)
        {
            const std::uint64_t transaction = loadSystemAcquire(
                &binding.controller->transaction_id);
            ready = binding.groupRoot() && transaction != 0u &&
                            waitForState(
                                binding,
                                MoEOverlayDeviceControllerState::
                                    CollectingSnapshots) &&
                            waitForAllGroupParticipants(
                                binding,
                                ParticipantWordField::SnapshotTransaction,
                                transaction)
                        ? 1u
                        : 0u;
            if (ready == 0u && binding.groupRoot())
                failGroup(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
        }
        __syncthreads();
        if (ready == 0u)
            return;
        const auto *group = groupLayout(binding, binding.group_id);
        if (!group)
        {
            if (threadIdx.x == 0u)
                failGroup(binding, MoEOverlayDeviceControllerError::InvalidGroup);
            return;
        }
        const std::uint64_t words =
            loadPeerPublished(&group->collected_state_words);
        std::uint64_t digest = 0u;
        std::uint64_t count = 0u;
        for (std::uint64_t index = threadIdx.x;
             index < words;
             index += blockDim.x)
        {
            const std::uint64_t word =
                loadPeerPublished(binding.group_collected_state + index);
            digest ^= digestWord(word, index);
            count = saturatingAdd(
                count,
                moe_rebalance_policy::collectedStateActivationCount(word));
        }
        digests[threadIdx.x] = digest;
        observations[threadIdx.x] = count;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2u;
             stride != 0u;
             stride >>= 1u)
        {
            if (threadIdx.x < stride)
            {
                digests[threadIdx.x] ^= digests[threadIdx.x + stride];
                observations[threadIdx.x] = saturatingAdd(
                    observations[threadIdx.x],
                    observations[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x != 0u)
            return;
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        const std::uint64_t prior =
            loadSystemAcquire(&binding.local_group->snapshot_transaction);
        if (transaction == 0u || prior > transaction)
        {
            failGroup(binding, MoEOverlayDeviceControllerError::InvalidTransaction);
            return;
        }
        binding.local_group->status_code =
            raw(MoEOverlayDeviceControllerError::None);
        binding.local_group->snapshot_digest =
            digests[0] == 0u ? 1u : digests[0];
        binding.local_group->snapshot_observations = observations[0];
        __threadfence_system();
        storeSystemRelease(
            &binding.local_group->snapshot_transaction, transaction);
    }

    /** Publish and independently validate the device-authored command batch. */
    __device__ __forceinline__ void publishCommand(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        __shared__ std::uint64_t policy_words[
            sizeof(MoEOverlayDeviceControllerPolicyResult) /
            sizeof(std::uint64_t)];
        __shared__ std::uint64_t digest_parts[kControllerThreads];
        __shared__ std::uint64_t byte_parts[kControllerThreads];
        __shared__ std::uint32_t invalid_parts[kControllerThreads];
        __shared__ std::uint32_t ready;
        const auto &binding = launch.binding;
        if (threadIdx.x == 0u)
        {
            const std::uint64_t transaction =
                loadSystemAcquire(&binding.controller->transaction_id);
            ready = binding.authorityLeader() && transaction != 0u &&
                            waitForState(
                                binding,
                                MoEOverlayDeviceControllerState::
                                    CollectingSnapshots) &&
                            waitForAllGroups(
                                binding,
                                GroupWordField::SnapshotTransaction,
                                transaction)
                        ? 1u
                        : 0u;
            if (ready == 0u && binding.authorityLeader())
            {
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
            }
            if (ready != 0u)
            {
                const auto snapshot = snapshotPeerRecord(
                    launch.policy_result);
                const auto *snapshot_words =
                    reinterpret_cast<const std::uint64_t *>(&snapshot);
                for (std::size_t lane = 0u;
                     lane < sizeof(policy_words) / sizeof(policy_words[0]);
                     ++lane)
                {
                    policy_words[lane] = snapshot_words[lane];
                }
            }
        }
        __syncthreads();
        if (ready == 0u)
            return;

        MoEOverlayDeviceControllerPolicyResult policy{};
        auto *local_policy_words = reinterpret_cast<std::uint64_t *>(
            &policy);
        for (std::size_t lane = 0u;
             lane < sizeof(policy_words) / sizeof(policy_words[0]);
             ++lane)
        {
            local_policy_words[lane] = policy_words[lane];
        }

        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        const auto kind = static_cast<
            MoEOverlayDeviceControllerTransactionKind>(policy.kind);
        const std::uint32_t capacity =
            loadPeerPublished(&binding.layout->command_capacity);
        const std::uint32_t participant_count =
            loadPeerPublished(&binding.layout->participant_count);
        const std::uint32_t num_layers =
            loadPeerPublished(&binding.layout->num_layers);
        const std::uint32_t num_experts =
            loadPeerPublished(&binding.layout->num_experts);

        // Authenticate cardinality before any thread indexes the mapped entry
        // array. A corrupt policy result must never turn validation into an
        // out-of-bounds peer-memory read.
        if (!policy.valid() || policy.command_count > capacity ||
            policy.kind !=
                loadSystemAcquire(&binding.controller->transaction_kind))
        {
            if (threadIdx.x == 0u)
            {
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidCommand);
            }
            return;
        }

        std::uint64_t digest = threadIdx.x == 0u
                                   ? moeOverlayCommandDigestSeed(
                                         policy.command_count)
                                   : 0u;
        std::uint64_t packed_bytes = 0u;
        std::uint32_t invalid = 0u;
        constexpr std::uint32_t kWordsPerCommand =
            sizeof(MoEOverlayDeviceMovementCommand) /
            sizeof(std::uint64_t);
        const std::uint64_t command_words =
            static_cast<std::uint64_t>(policy.command_count) *
            kWordsPerCommand;
        const auto *entry_words = reinterpret_cast<const std::uint64_t *>(
            binding.command_entries);
        for (std::uint64_t word_index = threadIdx.x;
             word_index < command_words;
             word_index += blockDim.x)
        {
            digest ^= moeOverlayCommandDigestWord(
                loadPeerPublished(entry_words + word_index), word_index);
        }
        for (std::uint32_t entry_index = threadIdx.x;
             entry_index < policy.command_count;
             entry_index += blockDim.x)
        {
            const auto entry = snapshotPeerRecord(
                binding.command_entries + entry_index);
            if (!moeOverlayMovementCommandValid(
                    entry,
                    entry_index,
                    kind,
                    participant_count,
                    num_layers,
                    num_experts,
                    binding.controller->base_epoch,
                    binding.controller->candidate_epoch))
            {
                ++invalid;
            }
            packed_bytes = saturatingAdd(
                packed_bytes, entry.payload_bytes);
            if (entry_index != 0u)
            {
                const auto previous = snapshotPeerRecord(
                    binding.command_entries + entry_index - 1u);
                if (!moeOverlayIndependentCanonicalDestination(
                        previous, entry))
                    ++invalid;
            }
        }

        digest_parts[threadIdx.x] = digest;
        byte_parts[threadIdx.x] = packed_bytes;
        invalid_parts[threadIdx.x] = invalid;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2u;
             stride != 0u;
             stride >>= 1u)
        {
            if (threadIdx.x < stride)
            {
                digest_parts[threadIdx.x] ^=
                    digest_parts[threadIdx.x + stride];
                byte_parts[threadIdx.x] = saturatingAdd(
                    byte_parts[threadIdx.x],
                    byte_parts[threadIdx.x + stride]);
                invalid_parts[threadIdx.x] +=
                    invalid_parts[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x != 0u)
            return;

        const bool semantic_result =
            invalid_parts[0] == 0u &&
            digest_parts[0] == policy.command_digest &&
            byte_parts[0] == policy.packed_weight_bytes &&
            ((kind ==
                  MoEOverlayDeviceControllerTransactionKind::StaticCheck &&
              policy.command_count == 0u &&
              policy.packed_weight_bytes == 0u) ||
             (kind == MoEOverlayDeviceControllerTransactionKind::
                          DynamicPlacement &&
              ((policy.command_count == 0u &&
                policy.packed_weight_bytes == 0u) ||
               (policy.command_count != 0u &&
                policy.packed_weight_bytes != 0u))) ||
             (kind ==
                  MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP &&
              policy.command_count != 0u));
        if (!semantic_result)
        {
            failLeader(binding, MoEOverlayDeviceControllerError::InvalidCommand);
            return;
        }

        binding.command->kind = policy.kind;
        binding.command->command_count = policy.command_count;
        binding.command->topology_fingerprint = binding.topology_fingerprint;
        binding.command->transaction_id = transaction;
        binding.command->base_epoch = binding.controller->base_epoch;
        binding.command->candidate_epoch = binding.controller->candidate_epoch;
        binding.command->command_digest = policy.command_digest;
        binding.command->packed_weight_bytes = policy.packed_weight_bytes;
        binding.command->parallel_command_count = policy.command_count;
        binding.command->movement_round_count =
            policy.command_count == 0u ? 0u : 1u;
        binding.command->hazard_count = 0u;
        binding.command->snapshot_observations =
            policy.snapshot_observations;
        binding.command->priority_cost_before =
            policy.priority_cost_before;
        binding.command->priority_cost_after =
            policy.priority_cost_after;
        binding.command->same_priority_makespan_before =
            policy.same_priority_makespan_before;
        binding.command->same_priority_makespan_after =
            policy.same_priority_makespan_after;
        binding.command->accepted_cycles = policy.accepted_cycles;
        binding.command->rejected_cycles = policy.rejected_cycles;
        binding.command->promotions = policy.promotions;
        binding.command->demotions = policy.demotions;
        binding.command->same_priority_moves =
            policy.same_priority_moves;
        binding.command->changed_layers = policy.changed_layers;
        binding.command->layer_scan_start = policy.layer_scan_start;
        binding.command->layer_scan_next = policy.layer_scan_next;
        binding.command->projected_service_gain_ns =
            policy.projected_service_gain_ns;
        binding.command->projected_transfer_and_repack_ns =
            policy.projected_transfer_and_repack_ns;
        binding.command->projected_inference_interference_ns =
            policy.projected_inference_interference_ns;
        binding.command->projected_net_benefit_ns =
            policy.projected_net_benefit_ns;
        binding.command->payoff_rejected_cycles =
            policy.payoff_rejected_cycles;
        binding.command->residency_rejected_cycles =
            policy.residency_rejected_cycles;
        __threadfence_system();
        storeSystemRelease(
            &binding.controller->command_transaction, transaction);
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::PreparingFollowers));
    }

    /**
     * @brief Author the immutable no-movement result for Static/Observe mode.
     *
     * Static policy is still a device decision: the leader validates that the
     * live transaction is a StaticCheck and writes the same persistent result
     * consumed by PublishCommand. Keeping this one cache-line write inside the
     * retained graph prevents setup code from manufacturing a host policy
     * record merely because the correct command cardinality happens to be zero.
     */
    __device__ __forceinline__ void authorStaticPolicy(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.authorityLeader() || !launch.policy_result ||
            !waitForState(
                binding,
                MoEOverlayDeviceControllerState::CollectingSnapshots) ||
            loadSystemAcquire(&binding.controller->transaction_kind) !=
                raw(MoEOverlayDeviceControllerTransactionKind::StaticCheck))
        {
            if (binding.authorityLeader())
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
            return;
        }

        MoEOverlayDeviceControllerPolicyResult result{};
        result.kind = raw(
            MoEOverlayDeviceControllerTransactionKind::StaticCheck);
        result.command_count = 0u;
        result.command_digest = moeOverlayCommandDigestSeed(0u);
        result.packed_weight_bytes = 0u;
        *launch.policy_result = result;
        __threadfence_system();
    }

    /** @return Canonical destination/layer/expert order for one command pair. */
    __device__ __forceinline__ bool dynamicCommandLess(
        const MoEOverlayDeviceMovementCommand &left,
        const MoEOverlayDeviceMovementCommand &right) noexcept
    {
        if (left.destination_participant != right.destination_participant)
        {
            return left.destination_participant <
                   right.destination_participant;
        }
        if (left.layer != right.layer)
            return left.layer < right.layer;
        return left.expert < right.expert;
    }

    /**
     * @brief Author one bounded two-axis Dynamic placement entirely on device.
     *
     * The leader consumes release-published participant snapshots, preserves
     * every participant's per-layer cardinality, accumulates the latest
     * phase-pure delta into leader-owned model-lifetime demand, assigns the
     * hottest experts across the combined prefill/decode history to the lowest
     * integer priorities, and greedily reduces makespan among equal-priority
     * participants. Only complete strictly improving cycles are emitted, so
     * every command can prepare concurrently and a partial wave cannot alter
     * capacity. One deterministic control thread owns both history and policy;
     * the host can neither mix phases nor mirror either value.
     */
    __device__ __forceinline__ void authorDynamicPolicy(
        const MoEOverlayDeviceControllerActionLaunch &launch,
        DynamicPolicyScratch &scratch) noexcept
    {
        const auto &binding = launch.binding;
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        if (!binding.authorityLeader() || !launch.policy_result ||
            transaction == 0u ||
            !waitForState(
                binding,
                MoEOverlayDeviceControllerState::CollectingSnapshots) ||
            !waitForAllGroups(
                binding,
                GroupWordField::SnapshotTransaction,
                transaction) ||
            loadSystemAcquire(&binding.controller->transaction_kind) !=
                raw(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement) ||
            loadPeerPublished(&binding.command->demand_phase) !=
                raw(launch.demand_phase))
        {
            if (binding.authorityLeader())
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
            return;
        }

        const std::uint32_t participant_count =
            loadPeerPublished(&binding.layout->participant_count);
        const std::uint32_t layer_count =
            loadPeerPublished(&binding.layout->num_layers);
        const std::uint32_t expert_count =
            loadPeerPublished(&binding.layout->num_experts);
        const std::uint32_t command_capacity =
            loadPeerPublished(&binding.layout->command_capacity);
        const std::uint32_t maximum_cycles =
            loadPeerPublished(&binding.layout->maximum_cycles_per_wave);
        const std::uint32_t imbalance_threshold_per_mille =
            loadPeerPublished(
                &binding.layout->dynamic_imbalance_threshold_per_mille);
        const std::uint32_t minimum_improvement_per_mille =
            loadPeerPublished(
                &binding.layout->dynamic_minimum_improvement_per_mille);
        const std::uint32_t maximum_cycles_per_layer =
            loadPeerPublished(
                &binding.layout->dynamic_maximum_cycles_per_layer);
        const std::uint32_t configured_maximum_commands =
            loadPeerPublished(
                &binding.layout->dynamic_maximum_commands_per_wave);
        const std::uint64_t minimum_observations =
            loadPeerPublished(&binding.layout->minimum_window_activations);
        const std::uint32_t economy_state = loadSystemAcquire(
            &binding.economy->state);
        const bool economy_ready =
            economy_state == raw(
                MoEOverlayDeviceControllerEconomyState::Ready);
        const std::uint64_t expected_history_words =
            static_cast<std::uint64_t>(
                kMoEOverlayDeviceControllerDemandPhaseCount) *
            layer_count * expert_count;
        const std::uint32_t demand_phase =
            moeOverlayDeviceDemandPhaseIndex(launch.demand_phase);
        if (participant_count == 0u ||
            participant_count >
                kMoEOverlayDeviceControllerFabricMaxParticipants ||
            layer_count == 0u ||
            layer_count > kMoEOverlayDeviceControllerFabricMaxLayers ||
            expert_count == 0u ||
            expert_count > kMoEOverlayDeviceControllerFabricMaxExperts ||
            binding.layout->tier_count == 0u ||
            binding.layout->tier_count >
                kMoEOverlayDeviceControllerFabricMaxParticipants ||
            command_capacity == 0u || maximum_cycles == 0u ||
            minimum_observations == 0u || !binding.demand_history ||
            demand_phase >= kMoEOverlayDeviceControllerDemandPhaseCount ||
            loadPeerPublished(&binding.layout->demand_history_words) !=
                expected_history_words ||
            (economy_state != raw(
                 MoEOverlayDeviceControllerEconomyState::Empty) &&
             !economy_ready) ||
            (economy_ready &&
             (loadPeerPublished(&binding.economy->magic) !=
                  kMoEOverlayDeviceControllerFabricMagic ||
              loadPeerPublished(&binding.economy->version) !=
                  kMoEOverlayDeviceControllerFabricVersion ||
              loadPeerPublished(&binding.economy->tier_count) !=
                  binding.layout->tier_count ||
              loadPeerPublished(&binding.economy->participant_count) !=
                  participant_count ||
              loadPeerPublished(&binding.economy->layer_count) !=
                  layer_count ||
              loadPeerPublished(&binding.economy->service_phase_count) !=
                  kMoEOverlayDeviceControllerEconomyServicePhaseCount ||
              loadPeerPublished(
                  &binding.economy->topology_fingerprint) !=
                  binding.topology_fingerprint ||
              loadPeerPublished(
                  &binding.economy->service_identity_fingerprint) == 0u ||
              loadPeerPublished(
                  &binding.economy->migration_identity_fingerprint) == 0u ||
              loadPeerPublished(
                  &binding.economy->publication_generation) == 0u ||
              (loadPeerPublished(
                   &binding.economy->active_source_bits) &
               0x3u) != 0x3u ||
              loadPeerPublished(
                  &binding.economy->current_window_weight) == 0u ||
              loadPeerPublished(
                  &binding.economy->payoff_horizon_tokens) == 0u)))
        {
            failLeader(
                binding,
                MoEOverlayDeviceControllerError::InvalidTopology);
            return;
        }

        std::uint32_t priority_count = 0u;
        for (std::uint32_t participant = 0u;
             participant < participant_count;
             ++participant)
        {
            const auto metadata = snapshotPeerRecord(
                binding.participants + participant);
            if (metadata.participant_id != participant ||
                metadata.group_id >= binding.layout->group_count ||
                metadata.tier_index < 0 ||
                static_cast<std::uint32_t>(metadata.tier_index) >=
                    binding.layout->tier_count)
            {
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidTopology);
                return;
            }
            const std::int32_t value = metadata.tier_priority;
            scratch.participant_priority[participant] = value;
            scratch.participant_tier[participant] = metadata.tier_index;
            scratch.participant_state_base[participant] =
                participantStateBase(binding, participant, metadata);
            if (!scratch.participant_state_base[participant])
            {
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidTopology);
                return;
            }
            std::uint32_t position = 0u;
            while (position < priority_count &&
                   scratch.priorities[position] < value)
            {
                ++position;
            }
            if (position < priority_count &&
                scratch.priorities[position] == value)
            {
                continue;
            }
            for (std::uint32_t index = priority_count;
                 index > position;
                 --index)
            {
                scratch.priorities[index] =
                    scratch.priorities[index - 1u];
            }
            scratch.priorities[position] = value;
            ++priority_count;
        }

        MoEOverlayDeviceControllerPolicyResult result{};
        result.kind = raw(
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement);
        const std::uint64_t base_epoch =
            loadSystemAcquire(&binding.controller->base_epoch);
        const std::uint64_t candidate_epoch =
            loadSystemAcquire(&binding.controller->candidate_epoch);
        const std::uint32_t maximum_commands =
            configured_maximum_commands == 0u ||
                    configured_maximum_commands > command_capacity
                ? command_capacity
                : configured_maximum_commands;
        const std::uint32_t layer_scan_start = loadSystemAcquire(
            &binding.controller->dynamic_layer_cursor);
        if (layer_scan_start >= layer_count)
        {
            failLeader(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        result.layer_scan_start = layer_scan_start;
        result.layer_scan_next = (layer_scan_start + 1u) % layer_count;

        for (std::uint32_t layer_offset = 0u;
             layer_offset < layer_count;
             ++layer_offset)
        {
            const std::uint32_t layer =
                (layer_scan_start + layer_offset) % layer_count;
            std::uint64_t observations = 0u;
            for (std::uint32_t participant = 0u;
                 participant < participant_count;
                 ++participant)
            {
                scratch.quotas[participant] = 0u;
            }
            for (std::uint32_t expert = 0u;
                 expert < expert_count;
                 ++expert)
            {
                std::uint64_t current_phase_count = 0u;
                std::uint32_t owner_count = 0u;
                std::int32_t owner = -1;
                for (std::uint32_t participant = 0u;
                     participant < participant_count;
                     ++participant)
                {
                    const std::uint64_t word = loadPeerPublished(
                        scratch.participant_state_base[participant] +
                        static_cast<std::uint64_t>(layer) * expert_count +
                        expert);
                    current_phase_count = saturatingAdd(
                        current_phase_count,
                        moe_rebalance_policy::
                            collectedStateActivationCount(word));
                    if (moe_rebalance_policy::
                            collectedStateAuthoritativeOwner(word))
                    {
                        if (!moe_rebalance_policy::
                                collectedStatePhysicallyResident(word))
                        {
                            failLeader(
                                binding,
                                MoEOverlayDeviceControllerError::InvalidSnapshot);
                            return;
                        }
                        owner = static_cast<std::int32_t>(participant);
                        ++owner_count;
                    }
                }
                if (owner_count != 1u)
                {
                    failLeader(
                        binding,
                        MoEOverlayDeviceControllerError::InvalidSnapshot);
                    return;
                }
                const std::uint64_t selected_history_offset =
                    moeOverlayDeviceDemandHistoryOffset(
                        demand_phase,
                        layer,
                        expert,
                        layer_count,
                        expert_count);
                const std::uint32_t other_phase = 1u - demand_phase;
                const std::uint64_t other_history_offset =
                    moeOverlayDeviceDemandHistoryOffset(
                        other_phase,
                        layer,
                        expert,
                        layer_count,
                        expert_count);
                const std::uint64_t selected_history = loadPeerPublished(
                    binding.demand_history + selected_history_offset);
                const std::uint64_t other_history = loadPeerPublished(
                    binding.demand_history + other_history_offset);
                const auto history_update =
                    moeOverlayAccumulateDeviceDemandHistory(
                        selected_history,
                        other_history,
                        current_phase_count);
                storeSystemRelease(
                    binding.demand_history + selected_history_offset,
                    history_update.updated_phase_count);

                /* Use the just-authored combined value directly. A
                 * mapped load after a system release is unnecessary and can
                 * observe a stale host-page cache line on some PCIe paths. */
                scratch.expert_counts[expert] =
                    history_update.combined_count;
                const bool selected_is_prefill = demand_phase ==
                    moeOverlayDeviceDemandPhaseIndex(
                        MoEOverlayDeviceDemandPhase::Prefill);
                scratch.phase_expert_counts
                    [kMoEOverlayDeviceControllerEconomyDecodePhase][expert] =
                    selected_is_prefill
                        ? other_history
                        : history_update.updated_phase_count;
                scratch.phase_expert_counts
                    [kMoEOverlayDeviceControllerEconomyPrefillPhase][expert] =
                    selected_is_prefill
                        ? history_update.updated_phase_count
                        : other_history;
                scratch.current_owner[expert] = owner;
                scratch.desired_owner[expert] = -1;
                scratch.excluded[expert] = 0u;
                ++scratch.quotas[static_cast<std::uint32_t>(owner)];
                observations = saturatingAdd(
                    observations, current_phase_count);
            }
            result.snapshot_observations = saturatingAdd(
                result.snapshot_observations, observations);
            /* Before setup calibration is complete, this transaction remains
             * an observe-only device epoch. Histories still accumulate, so
             * the first certified decision sees every warmup/prefill sample;
             * the host cannot substitute an unmeasured placement meanwhile. */
            if (!economy_ready || observations < minimum_observations)
                continue;

            /* The economy profile is immutable for the transaction, but its
             * publication storage is node-local mapped memory. Cache the
             * current layer once before enumerating thousands of candidate
             * cycles; all later deterministic arithmetic stays in LDS/shared
             * memory and cannot amplify PCIe latency. */
            for (std::uint32_t tier = 0u;
                 tier < binding.layout->tier_count;
                 ++tier)
            {
                for (std::uint32_t phase_index = 0u;
                     phase_index <
                         kMoEOverlayDeviceControllerDemandPhaseCount;
                     ++phase_index)
                {
                    scratch.service_cost[tier][phase_index] =
                        dynamicEconomyServiceCost(
                            binding, tier, layer, phase_index);
                }
            }
            for (std::uint32_t source = 0u;
                 source < participant_count;
                 ++source)
            {
                for (std::uint32_t destination = 0u;
                     destination < participant_count;
                     ++destination)
                {
                    const auto movement = dynamicEconomyMigrationCost(
                        binding, source, destination, layer);
                    scratch.migration_transfer_and_repack_ns
                        [source][destination] =
                            movement.transfer_and_repack_ns;
                    scratch.migration_inference_interference_ns
                        [source][destination] =
                            movement.inference_interference_ns;
                }
            }
            for (std::uint32_t expert = 0u;
                 expert < expert_count;
                 ++expert)
            {
                scratch.last_moved[expert] = loadPeerPublished(
                    binding.economy_last_moved +
                    static_cast<std::uint64_t>(layer) * expert_count +
                    expert);
            }
            scratch.minimum_residency_generations = loadPeerPublished(
                &binding.economy->minimum_residency_generations);
            scratch.payoff_horizon_tokens = loadPeerPublished(
                &binding.economy->payoff_horizon_tokens);
            scratch.minimum_net_benefit_ns = loadPeerPublished(
                &binding.economy->minimum_net_benefit_ns);

            // Selection sort is intentionally explicit: CUDA, HIP, and the
            // CPU oracle all use count-descending/expert-ascending order.
            for (std::uint32_t expert = 0u;
                 expert < expert_count;
                 ++expert)
            {
                scratch.ordered_experts[expert] = expert;
            }
            for (std::uint32_t position = 0u;
                 position < expert_count;
                 ++position)
            {
                std::uint32_t selected = position;
                for (std::uint32_t candidate = position + 1u;
                     candidate < expert_count;
                     ++candidate)
                {
                    const std::uint32_t left =
                        scratch.ordered_experts[candidate];
                    const std::uint32_t right =
                        scratch.ordered_experts[selected];
                    if (scratch.expert_counts[left] >
                            scratch.expert_counts[right] ||
                        (scratch.expert_counts[left] ==
                             scratch.expert_counts[right] &&
                         left < right))
                    {
                        selected = candidate;
                    }
                }
                const std::uint32_t temporary =
                    scratch.ordered_experts[position];
                scratch.ordered_experts[position] =
                    scratch.ordered_experts[selected];
                scratch.ordered_experts[selected] = temporary;
            }

            std::uint32_t expert_cursor = 0u;
            for (std::uint32_t priority_index = 0u;
                 priority_index < priority_count;
                 ++priority_index)
            {
                std::uint32_t member_count = 0u;
                std::uint32_t group_quota = 0u;
                for (std::uint32_t participant = 0u;
                     participant < participant_count;
                     ++participant)
                {
                    if (scratch.participant_priority[participant] ==
                        scratch.priorities[priority_index])
                    {
                        scratch.priority_participants[member_count] =
                            participant;
                        scratch.remaining[member_count] =
                            scratch.quotas[participant];
                        scratch.participant_load[member_count] = 0u;
                        group_quota += scratch.quotas[participant];
                        ++member_count;
                    }
                }
                for (std::uint32_t slot = 0u;
                     slot < group_quota;
                     ++slot)
                {
                    if (expert_cursor >= expert_count)
                    {
                        failLeader(
                            binding,
                            MoEOverlayDeviceControllerError::InvalidSnapshot);
                        return;
                    }
                    const std::uint32_t expert =
                        scratch.ordered_experts[expert_cursor++];
                    std::uint32_t selected = member_count;
                    for (std::uint32_t member = 0u;
                         member < member_count;
                         ++member)
                    {
                        if (scratch.remaining[member] == 0u)
                            continue;
                        if (selected == member_count ||
                            scratch.participant_load[member] <
                                scratch.participant_load[selected] ||
                            (scratch.participant_load[member] ==
                                 scratch.participant_load[selected] &&
                             scratch.priority_participants[member] <
                                 scratch.priority_participants[selected]))
                        {
                            selected = member;
                        }
                    }
                    if (selected == member_count)
                    {
                        failLeader(
                            binding,
                            MoEOverlayDeviceControllerError::InvalidSnapshot);
                        return;
                    }
                    scratch.desired_owner[expert] =
                        static_cast<std::int32_t>(
                            scratch.priority_participants[selected]);
                    --scratch.remaining[selected];
                    scratch.participant_load[selected] = saturatingAdd(
                        scratch.participant_load[selected],
                        scratch.expert_counts[expert]);
                }
            }
            if (expert_cursor != expert_count)
            {
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidSnapshot);
                return;
            }

            DynamicPlacementScore working_score = scoreDynamicLayer(
                scratch,
                scratch.current_owner,
                participant_count,
                expert_count,
                priority_count);
            result.priority_cost_before = saturatingAdd(
                result.priority_cost_before,
                working_score.priority_cost);
            result.same_priority_makespan_before = saturatingAdd(
                result.same_priority_makespan_before,
                working_score.same_priority_makespan);

            bool changed_layer = false;
            std::uint32_t layer_cycles = 0u;
            while (result.accepted_cycles < maximum_cycles &&
                   layer_cycles < maximum_cycles_per_layer)
            {
                /* Decompose the remaining owner/target multigraph into its
                 * canonical disjoint cycles, then admit the best measured
                 * payoff. Expert-id order is only the final deterministic
                 * tie break; it is no longer an accidental economy policy. */
                for (std::uint32_t expert = 0u;
                     expert < expert_count;
                     ++expert)
                {
                    scratch.enumerated[expert] = scratch.excluded[expert];
                }
                std::uint32_t best_cycle_length = 0u;
                DynamicCycleEconomyScore best_economy{};
                while (true)
                {
                    const std::uint32_t cycle_length = findDynamicCycle(
                        scratch,
                        scratch.enumerated,
                        participant_count,
                        expert_count);
                    if (cycle_length == 0u)
                        break;
                    for (std::uint32_t edge = 0u;
                         edge < cycle_length;
                         ++edge)
                    {
                        scratch.enumerated[scratch.cycle[edge]] = 1u;
                    }
                    if (result.command_count + cycle_length > maximum_commands)
                        continue;

                    for (std::uint32_t expert = 0u;
                         expert < expert_count;
                         ++expert)
                    {
                        scratch.candidate_owner[expert] =
                            scratch.current_owner[expert];
                    }
                    for (std::uint32_t edge = 0u;
                         edge < cycle_length;
                         ++edge)
                    {
                        const std::uint32_t expert = scratch.cycle[edge];
                        scratch.candidate_owner[expert] =
                            scratch.desired_owner[expert];
                    }
                    const DynamicPlacementScore candidate_score =
                        scoreDynamicLayer(
                            scratch,
                            scratch.candidate_owner,
                            participant_count,
                            expert_count,
                            priority_count);
                    const bool improves_priority =
                        candidate_score.priority_cost <
                        working_score.priority_cost;
                    const bool targets_skew =
                        !improves_priority &&
                        samePriorityDynamicCycleIsEligible(
                            scratch,
                            cycle_length,
                            participant_count,
                            expert_count,
                            imbalance_threshold_per_mille);
                    if (!improves_priority && !targets_skew)
                    {
                        ++result.rejected_cycles;
                        for (std::uint32_t edge = 0u;
                             edge < cycle_length;
                             ++edge)
                        {
                            scratch.excluded[scratch.cycle[edge]] = 1u;
                        }
                        continue;
                    }

                    const DynamicCycleEconomyScore economy =
                        scoreDynamicCycleEconomy(
                            binding,
                            scratch,
                            cycle_length,
                            layer,
                            participant_count,
                            expert_count,
                            transaction,
                            minimum_improvement_per_mille);
                    result.residency_rejected_cycles +=
                        economy.residency_eligible ? 0u : 1u;
                    result.payoff_rejected_cycles +=
                        economy.payoff_eligible ? 0u : 1u;
                    if (!economy.eligible())
                    {
                        ++result.rejected_cycles;
                        /* Reconsider the cycle in a later transaction after
                         * demand, prices, or hysteresis age changes. Within
                         * this bounded wave it cannot become admissible until
                         * another cycle changes placement, so exclude it. */
                        for (std::uint32_t edge = 0u;
                             edge < cycle_length;
                             ++edge)
                        {
                            scratch.excluded[scratch.cycle[edge]] = 1u;
                        }
                        continue;
                    }
                    if (best_cycle_length == 0u ||
                        dynamicEconomyScoreBetter(economy, best_economy))
                    {
                        best_cycle_length = cycle_length;
                        best_economy = economy;
                        for (std::uint32_t edge = 0u;
                             edge < cycle_length;
                             ++edge)
                        {
                            scratch.best_cycle[edge] = scratch.cycle[edge];
                        }
                    }
                }

                if (best_cycle_length == 0u)
                    break;
                for (std::uint32_t edge = 0u;
                     edge < best_cycle_length;
                     ++edge)
                {
                    scratch.cycle[edge] = scratch.best_cycle[edge];
                }
                for (std::uint32_t expert = 0u;
                     expert < expert_count;
                     ++expert)
                {
                    scratch.candidate_owner[expert] =
                        scratch.current_owner[expert];
                }
                for (std::uint32_t edge = 0u;
                     edge < best_cycle_length;
                     ++edge)
                {
                    const std::uint32_t expert = scratch.cycle[edge];
                    scratch.candidate_owner[expert] =
                        scratch.desired_owner[expert];
                }
                const DynamicPlacementScore candidate_score =
                    scoreDynamicLayer(
                        scratch,
                        scratch.candidate_owner,
                        participant_count,
                        expert_count,
                        priority_count);
                const std::uint64_t payload_bytes = loadPeerPublished(
                    binding.payload_bytes_per_layer + layer);
                if (payload_bytes == 0u)
                {
                    failLeader(
                        binding,
                        MoEOverlayDeviceControllerError::InvalidControl);
                    return;
                }
                for (std::uint32_t edge = 0u;
                     edge < best_cycle_length;
                     ++edge)
                {
                    const std::uint32_t expert = scratch.cycle[edge];
                    const auto source = static_cast<std::uint32_t>(
                        scratch.current_owner[expert]);
                    const auto destination = static_cast<std::uint32_t>(
                        scratch.desired_owner[expert]);
                    MoEOverlayDeviceMovementCommand command{};
                    command.op = raw(
                        MoEOverlayDeviceMovementOp::DurableMove);
                    command.layer = layer;
                    command.expert = expert;
                    command.source_participant = source;
                    command.destination_participant = destination;
                    command.payload_bytes = payload_bytes;
                    command.source_epoch = base_epoch;
                    command.candidate_epoch = candidate_epoch;
                    binding.command_entries[result.command_count++] = command;

                    const std::int32_t source_priority =
                        scratch.participant_priority[source];
                    const std::int32_t destination_priority =
                        scratch.participant_priority[destination];
                    if (destination_priority < source_priority)
                        ++result.promotions;
                    else if (destination_priority > source_priority)
                        ++result.demotions;
                    else
                        ++result.same_priority_moves;
                }
                for (std::uint32_t expert = 0u;
                     expert < expert_count;
                     ++expert)
                {
                    scratch.current_owner[expert] =
                        scratch.candidate_owner[expert];
                }
                working_score = candidate_score;
                result.projected_service_gain_ns = saturatingAdd(
                    result.projected_service_gain_ns,
                    best_economy.projected_service_gain_ns);
                result.projected_transfer_and_repack_ns = saturatingAdd(
                    result.projected_transfer_and_repack_ns,
                    best_economy.transfer_and_repack_ns);
                result.projected_inference_interference_ns = saturatingAdd(
                    result.projected_inference_interference_ns,
                    best_economy.inference_interference_ns);
                result.projected_net_benefit_ns = saturatingAdd(
                    result.projected_net_benefit_ns,
                    best_economy.projected_net_benefit_ns);
                ++result.accepted_cycles;
                ++layer_cycles;
                changed_layer = true;
            }
            if (changed_layer)
            {
                ++result.changed_layers;
                result.layer_scan_next = (layer + 1u) % layer_count;
            }
            result.priority_cost_after = saturatingAdd(
                result.priority_cost_after,
                working_score.priority_cost);
            result.same_priority_makespan_after = saturatingAdd(
                result.same_priority_makespan_after,
                working_score.same_priority_makespan);
        }

        // Canonicalize after every layer contributes. This is the immutable
        // order consumed independently by every destination prepare graph.
        for (std::uint32_t position = 1u;
             position < result.command_count;
             ++position)
        {
            const auto value = binding.command_entries[position];
            std::uint32_t cursor = position;
            while (cursor != 0u && dynamicCommandLess(
                       value, binding.command_entries[cursor - 1u]))
            {
                binding.command_entries[cursor] =
                    binding.command_entries[cursor - 1u];
                --cursor;
            }
            binding.command_entries[cursor] = value;
        }

        result.command_digest = moeOverlayCommandDigestSeed(
            result.command_count);
        constexpr std::uint32_t kWordsPerCommand =
            sizeof(MoEOverlayDeviceMovementCommand) /
            sizeof(std::uint64_t);
        for (std::uint32_t command = 0u;
             command < result.command_count;
             ++command)
        {
            binding.command_entries[command].ordinal = command;
            binding.command_entries[command].payload_slot = command;
            result.packed_weight_bytes = saturatingAdd(
                result.packed_weight_bytes,
                binding.command_entries[command].payload_bytes);
        }
        const auto *command_words = reinterpret_cast<const std::uint64_t *>(
            binding.command_entries);
        const std::uint64_t word_count =
            static_cast<std::uint64_t>(result.command_count) *
            kWordsPerCommand;
        for (std::uint64_t word = 0u; word < word_count; ++word)
        {
            result.command_digest ^= moeOverlayCommandDigestWord(
                command_words[word], word);
        }

        // A no-op observation does not manufacture a new durable epoch. The
        // remaining protocol still publishes zero-movement evidence to every
        // group, preserving one lifecycle for Static and Dynamic observation.
        if (result.command_count == 0u)
            binding.controller->candidate_epoch = base_epoch;
        // Persist policy fairness on the device before publishing the result.
        // The next transaction reads this cursor; no host shadow is involved.
        storeSystemRelease(
            &binding.controller->dynamic_layer_cursor,
            result.layer_scan_next);
        __threadfence_system();
        *launch.policy_result = result;
        __threadfence_system();
    }

    /** Publish one complete participant-local apply result with code last. */
    __device__ __forceinline__ void finishRuntimeApplyStatus(
        MoEOverlayDeviceRuntimeApplyStatus *status,
        MoEOverlayDeviceRuntimeApplyCode code,
        std::uint64_t transaction,
        std::uint64_t base_epoch,
        std::uint64_t candidate_epoch,
        std::uint32_t candidate_bank,
        std::uint32_t commands_observed,
        std::uint32_t commands_applied,
        std::uint32_t changed_layers,
        std::uint32_t missing_arrivals,
        std::uint32_t invalid_runtime_layers) noexcept
    {
        status->transaction_id = transaction;
        status->base_epoch = base_epoch;
        status->candidate_epoch = candidate_epoch;
        status->candidate_bank = candidate_bank;
        status->commands_observed = commands_observed;
        status->commands_applied = commands_applied;
        status->changed_layers = changed_layers;
        status->missing_arrivals = missing_arrivals;
        status->invalid_runtime_layers = invalid_runtime_layers;
        status->reserved0 = 0u;
        status->reserved[0] = 0u;
        __threadfence();
        storeSystemRelease(&status->code, raw(code));
    }

    /** @return One coherent device-scope load of a local 64-bit RCU word. */
    __device__ __forceinline__ std::uint64_t loadLocalEpochWord(
        const std::uint64_t *address) noexcept
    {
        return static_cast<std::uint64_t>(atomicAdd(
            reinterpret_cast<unsigned long long *>(
                const_cast<std::uint64_t *>(address)),
            0ull));
    }

    /** @return One coherent device-scope load of a local 32-bit RCU word. */
    __device__ __forceinline__ std::uint32_t loadLocalEpochWord(
        const std::uint32_t *address) noexcept
    {
        return atomicAdd(
            reinterpret_cast<unsigned int *>(
                const_cast<std::uint32_t *>(address)),
            0u);
    }

    /** @brief Atomically publish one local 64-bit RCU word. */
    __device__ __forceinline__ void storeLocalEpochWord(
        std::uint64_t *address,
        std::uint64_t value) noexcept
    {
        atomicExch(
            reinterpret_cast<unsigned long long *>(address),
            static_cast<unsigned long long>(value));
    }

    /** @brief Atomically publish one local 32-bit RCU word. */
    __device__ __forceinline__ void storeLocalEpochWord(
        std::uint32_t *address,
        std::uint32_t value) noexcept
    {
        atomicExch(reinterpret_cast<unsigned int *>(address), value);
    }

    /** @return Prior lifecycle word from one local device-scope CAS. */
    __device__ __forceinline__ std::uint32_t compareExchangeLocalEpochState(
        std::uint32_t *address,
        std::uint32_t expected,
        std::uint32_t desired) noexcept
    {
        return atomicCAS(
            reinterpret_cast<unsigned int *>(address), expected, desired);
    }

    /** Begin one controller-owned epoch status; the code remains the release word. */
    __device__ __forceinline__ void beginLocalEpochStatus(
        DeviceMoEOverlayEpochStatus *status,
        DeviceMoEOverlayEpochOperation operation) noexcept
    {
        storeLocalEpochWord(
            &status->code,
            raw(DeviceMoEOverlayEpochStatusCode::Idle));
        status->epoch = 0u;
        status->selector = 0u;
        status->operation = raw(operation);
        status->bank = kDeviceMoEOverlayInvalidBank;
        status->observed_state =
            raw(DeviceMoEOverlayEpochBankState::Empty);
    }

    /** Release-publish one complete controller-owned epoch result. */
    __device__ __forceinline__ void finishLocalEpochStatus(
        DeviceMoEOverlayEpochStatus *status,
        DeviceMoEOverlayEpochOperation operation,
        DeviceMoEOverlayEpochStatusCode code,
        std::uint64_t epoch,
        std::uint64_t selector,
        std::uint32_t bank,
        std::uint32_t observed_state) noexcept
    {
        status->epoch = epoch;
        status->selector = selector;
        status->operation = raw(operation);
        status->bank = bank;
        status->observed_state = observed_state;
        __threadfence();
        storeLocalEpochWord(&status->code, raw(code));
    }

    /** @return Bank containing an exact local epoch, or the invalid sentinel. */
    __device__ __forceinline__ std::uint32_t findLocalEpochBank(
        const DeviceMoEOverlayEpochControl *control,
        std::uint64_t epoch) noexcept
    {
        if (epoch == 0u)
            return kDeviceMoEOverlayInvalidBank;
        for (std::uint32_t bank = 0u;
             bank < kDeviceMoEOverlayEpochBankCount;
             ++bank)
        {
            if (loadLocalEpochWord(&control->bank_epochs[bank]) == epoch)
                return bank;
        }
        return kDeviceMoEOverlayInvalidBank;
    }

    /**
     * @brief Reserve the inactive local bank after command/arrival preflight.
     *
     * This is deliberately part of ApplyRuntimeCandidate rather than an
     * independently captured kernel. A rejected topology transaction therefore
     * cannot reserve or publish an RCU bank merely because later graph nodes
     * remain unconditional on CUDA or HIP.
     */
    __device__ __forceinline__ bool reserveLocalEpochCandidate(
        DeviceMoEOverlayEpochControl *control,
        std::uint64_t candidate_epoch,
        DeviceMoEOverlayEpochStatus *status,
        std::uint32_t expected_candidate,
        std::uint32_t expected_published) noexcept
    {
        beginLocalEpochStatus(
            status, DeviceMoEOverlayEpochOperation::ReserveCandidate);
        const std::uint64_t selector =
            loadLocalEpochWord(&control->published_selector);
        const std::uint32_t published =
            static_cast<std::uint32_t>(selector & 1u);
        const std::uint64_t generation = selector >> 1u;
        if (generation == 0u || published >= kDeviceMoEOverlayEpochBankCount ||
            published != expected_published ||
            expected_candidate >= kDeviceMoEOverlayEpochBankCount ||
            expected_candidate == published)
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidControl,
                candidate_epoch,
                selector,
                kDeviceMoEOverlayInvalidBank,
                raw(DeviceMoEOverlayEpochBankState::Empty));
            return false;
        }

        const std::uint64_t published_epoch =
            loadLocalEpochWord(&control->bank_epochs[published]);
        if (candidate_epoch == 0u || candidate_epoch <= published_epoch ||
            loadLocalEpochWord(&control->bank_states[published]) !=
                raw(DeviceMoEOverlayEpochBankState::Published))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::InvalidArgument,
                candidate_epoch,
                selector,
                expected_candidate,
                loadLocalEpochWord(
                    &control->bank_states[expected_candidate]));
            return false;
        }

        std::uint32_t candidate_state =
            loadLocalEpochWord(&control->bank_states[expected_candidate]);
        if (candidate_state ==
                raw(DeviceMoEOverlayEpochBankState::Retiring) &&
            loadLocalEpochWord(&control->acquisitions_in_flight) == 0u &&
            loadLocalEpochWord(
                &control->bank_readers[expected_candidate]) == 0u)
        {
            const std::uint32_t retired = compareExchangeLocalEpochState(
                &control->bank_states[expected_candidate],
                raw(DeviceMoEOverlayEpochBankState::Retiring),
                raw(DeviceMoEOverlayEpochBankState::Empty));
            if (retired == raw(DeviceMoEOverlayEpochBankState::Retiring))
            {
                storeLocalEpochWord(
                    &control->bank_epochs[expected_candidate], 0u);
                candidate_state =
                    raw(DeviceMoEOverlayEpochBankState::Empty);
            }
            else
            {
                candidate_state = retired;
            }
        }
        if (loadLocalEpochWord(
                &control->bank_readers[expected_candidate]) != 0u ||
            loadLocalEpochWord(
                &control->bank_epochs[expected_candidate]) != 0u ||
            candidate_state != raw(DeviceMoEOverlayEpochBankState::Empty))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::Busy,
                candidate_epoch,
                selector,
                expected_candidate,
                candidate_state);
            return false;
        }

        const std::uint32_t prior = compareExchangeLocalEpochState(
            &control->bank_states[expected_candidate],
            raw(DeviceMoEOverlayEpochBankState::Empty),
            raw(DeviceMoEOverlayEpochBankState::Candidate));
        if (prior != raw(DeviceMoEOverlayEpochBankState::Empty))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::ReserveCandidate,
                DeviceMoEOverlayEpochStatusCode::Busy,
                candidate_epoch,
                selector,
                expected_candidate,
                prior);
            return false;
        }
        storeLocalEpochWord(
            &control->bank_epochs[expected_candidate], candidate_epoch);
        finishLocalEpochStatus(
            status,
            DeviceMoEOverlayEpochOperation::ReserveCandidate,
            DeviceMoEOverlayEpochStatusCode::Success,
            candidate_epoch,
            selector,
            expected_candidate,
            raw(DeviceMoEOverlayEpochBankState::Candidate));
        return true;
    }

    /**
     * @brief Mark and publish one preflighted local bank as one authority action.
     *
     * The controller commit is validated by the caller immediately before this
     * helper. If the Ready-to-Published CAS cannot complete, the first CAS is
     * rolled back to Candidate because the selector is still unchanged.
     */
    __device__ __forceinline__ bool publishLocalEpochCandidate(
        DeviceMoEOverlayEpochControl *control,
        std::uint64_t candidate_epoch,
        DeviceMoEOverlayEpochStatus *status,
        std::uint32_t candidate,
        std::uint32_t previous) noexcept
    {
        beginLocalEpochStatus(
            status, DeviceMoEOverlayEpochOperation::PublishCandidate);
        const std::uint64_t old_selector =
            loadLocalEpochWord(&control->published_selector);
        const std::uint64_t old_generation = old_selector >> 1u;
        if (candidate >= kDeviceMoEOverlayEpochBankCount ||
            previous >= kDeviceMoEOverlayEpochBankCount ||
            candidate == previous ||
            static_cast<std::uint32_t>(old_selector & 1u) != previous ||
            old_generation == 0u ||
            loadLocalEpochWord(&control->bank_epochs[candidate]) !=
                candidate_epoch ||
            loadLocalEpochWord(&control->bank_states[candidate]) !=
                raw(DeviceMoEOverlayEpochBankState::Candidate) ||
            loadLocalEpochWord(&control->bank_states[previous]) !=
                raw(DeviceMoEOverlayEpochBankState::Published))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                candidate_epoch,
                old_selector,
                candidate,
                candidate < kDeviceMoEOverlayEpochBankCount
                    ? loadLocalEpochWord(&control->bank_states[candidate])
                    : raw(DeviceMoEOverlayEpochBankState::Empty));
            return false;
        }
        if (old_generation == (~0ull >> 1u))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::GenerationOverflow,
                candidate_epoch,
                old_selector,
                candidate,
                loadLocalEpochWord(&control->bank_states[candidate]));
            return false;
        }

        const std::uint32_t marked = compareExchangeLocalEpochState(
            &control->bank_states[candidate],
            raw(DeviceMoEOverlayEpochBankState::Candidate),
            raw(DeviceMoEOverlayEpochBankState::Ready));
        if (marked != raw(DeviceMoEOverlayEpochBankState::Candidate))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                candidate_epoch,
                old_selector,
                candidate,
                marked);
            return false;
        }
        const std::uint32_t published = compareExchangeLocalEpochState(
            &control->bank_states[candidate],
            raw(DeviceMoEOverlayEpochBankState::Ready),
            raw(DeviceMoEOverlayEpochBankState::Published));
        if (published != raw(DeviceMoEOverlayEpochBankState::Ready))
        {
            (void)compareExchangeLocalEpochState(
                &control->bank_states[candidate],
                raw(DeviceMoEOverlayEpochBankState::Ready),
                raw(DeviceMoEOverlayEpochBankState::Candidate));
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::PublishCandidate,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                candidate_epoch,
                old_selector,
                candidate,
                published);
            return false;
        }

        // Every candidate descriptor write precedes the selector flip. The
        // inference-side external admission epoch remains on the base epoch
        // until all participants have published, so fan-out is still safe.
        __threadfence();
        const std::uint64_t selector =
            ((old_generation + 1u) << 1u) |
            static_cast<std::uint64_t>(candidate);
        storeLocalEpochWord(&control->published_selector, selector);
        storeLocalEpochWord(
            &control->bank_states[previous],
            raw(DeviceMoEOverlayEpochBankState::Retiring));
        finishLocalEpochStatus(
            status,
            DeviceMoEOverlayEpochOperation::PublishCandidate,
            DeviceMoEOverlayEpochStatusCode::Success,
            candidate_epoch,
            selector,
            candidate,
            raw(DeviceMoEOverlayEpochBankState::Published));
        return true;
    }

    /**
     * @brief Reclaim one bank after its topology-wide readiness receipt.
     *
     * The caller has already observed every participant's immutable
     * `retirement_ready_epoch`.  That receipt was published only after the
     * topology admission word selected the candidate epoch and the local
     * acquisition guard plus old-bank reader count were both zero.  Therefore
     * a later acquisition can target only the candidate bank.  Rechecking the
     * device-wide acquisition guard here would incorrectly couple reclamation
     * to unrelated new-epoch traffic and make concurrent inference fail a safe
     * retirement.  The old bank's reader count remains the exact local safety
     * invariant and is monotonic after certification.
     */
    __device__ __forceinline__ bool retireReadinessCertifiedLocalEpoch(
        DeviceMoEOverlayEpochControl *control,
        std::uint64_t retiring_epoch,
        DeviceMoEOverlayEpochStatus *status) noexcept
    {
        beginLocalEpochStatus(status, DeviceMoEOverlayEpochOperation::Retire);
        const std::uint32_t bank =
            findLocalEpochBank(control, retiring_epoch);
        const std::uint64_t selector =
            loadLocalEpochWord(&control->published_selector);
        if (bank >= kDeviceMoEOverlayEpochBankCount ||
            loadLocalEpochWord(&control->bank_states[bank]) !=
                raw(DeviceMoEOverlayEpochBankState::Retiring))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                DeviceMoEOverlayEpochStatusCode::NotReady,
                retiring_epoch,
                selector,
                bank,
                bank < kDeviceMoEOverlayEpochBankCount
                    ? loadLocalEpochWord(&control->bank_states[bank])
                    : raw(DeviceMoEOverlayEpochBankState::Empty));
            return false;
        }
        if (loadLocalEpochWord(&control->bank_readers[bank]) != 0u)
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                DeviceMoEOverlayEpochStatusCode::Busy,
                retiring_epoch,
                selector,
                bank,
                raw(DeviceMoEOverlayEpochBankState::Retiring));
            return false;
        }
        const std::uint32_t prior = compareExchangeLocalEpochState(
            &control->bank_states[bank],
            raw(DeviceMoEOverlayEpochBankState::Retiring),
            raw(DeviceMoEOverlayEpochBankState::Empty));
        if (prior != raw(DeviceMoEOverlayEpochBankState::Retiring))
        {
            finishLocalEpochStatus(
                status,
                DeviceMoEOverlayEpochOperation::Retire,
                DeviceMoEOverlayEpochStatusCode::Busy,
                retiring_epoch,
                selector,
                bank,
                prior);
            return false;
        }
        storeLocalEpochWord(&control->bank_epochs[bank], 0u);
        finishLocalEpochStatus(
            status,
            DeviceMoEOverlayEpochOperation::Retire,
            DeviceMoEOverlayEpochStatusCode::Success,
            retiring_epoch,
            selector,
            bank,
            raw(DeviceMoEOverlayEpochBankState::Empty));
        return true;
    }

    /** @return Whether one local retiring bank currently has no possible reader. */
    __device__ __forceinline__ bool localEpochRetirementReady(
        const DeviceMoEOverlayEpochControl *control,
        std::uint64_t retiring_epoch,
        std::uint32_t *retiring_bank) noexcept
    {
        if (!control || !retiring_bank)
            return false;
        const std::uint32_t bank = findLocalEpochBank(control, retiring_epoch);
        *retiring_bank = bank;
        return bank < kDeviceMoEOverlayEpochBankCount &&
               loadLocalEpochWord(&control->bank_states[bank]) ==
                   raw(DeviceMoEOverlayEpochBankState::Retiring) &&
               loadLocalEpochWord(&control->acquisitions_in_flight) == 0u &&
               loadLocalEpochWord(&control->bank_readers[bank]) == 0u;
    }

    /**
     * @return Whether transport has made every local arrival descriptor visible.
     *
     * Every participant waits on the same group-owned physical edge. A failure
     * is additionally copied into the participant lane because non-root GPUs
     * are not permitted to mutate the group acknowledgement record.
     */
    __device__ __forceinline__ bool waitForRuntimeTransport(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        std::uint64_t transaction,
        std::uint64_t command_digest) noexcept
    {
        while (true)
        {
            const std::uint32_t status =
                loadSystemAcquire(&binding.local_transport->status_code);
            if (status != raw(MoEOverlayDeviceControllerError::None))
            {
                failParticipant(
                    binding,
                    MoEOverlayDeviceControllerError::PhysicalTransportFailure);
                if (binding.groupRoot())
                {
                    failGroup(
                        binding,
                        MoEOverlayDeviceControllerError::
                            PhysicalTransportFailure);
                }
                return false;
            }
            const std::uint64_t observed_transaction = loadSystemAcquire(
                &binding.local_transport->command_transaction);
            if (observed_transaction > transaction)
            {
                failParticipant(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidTransaction);
                if (binding.groupRoot())
                {
                    failGroup(
                        binding,
                        MoEOverlayDeviceControllerError::InvalidTransaction);
                }
                return false;
            }
            if (observed_transaction == transaction &&
                loadPeerPublished(
                    &binding.local_transport->command_digest) ==
                    command_digest &&
                loadSystemAcquire(
                    &binding.local_transport->prepared_transaction) ==
                    transaction)
            {
                return true;
            }
            if (loadSystemAcquire(&binding.controller->state) ==
                raw(MoEOverlayDeviceControllerState::Error))
            {
                return false;
            }
            peerWaitBackoff();
        }
    }

    /** @return Whether immutable participant metadata names its group slot. */
    __device__ __forceinline__ bool participantMetadataValid(
        const MoEOverlayDeviceControllerDeviceBinding &binding,
        const MoEOverlayDeviceControllerParticipantMetadata &metadata,
        std::uint32_t participant) noexcept
    {
        const auto *group = groupLayout(binding, metadata.group_id);
        return metadata.participant_id == participant && group != nullptr &&
               metadata.domain_participant_index < group->participant_count &&
               loadPeerPublished(
                   &group->participant_ids[
                       metadata.domain_participant_index]) == participant;
    }

    /**
     * @brief Build one participant's complete inactive durable runtime bank.
     *
     * The command and every source runtime are validated before the first
     * candidate byte changes. All participants derive the same global route;
     * only the exact destination installs the transfer-prepared weight triple.
     * The local RCU selector is deliberately untouched. A later commit-phase
     * action publishes the already-complete bank after every participant has
     * acknowledged preparation.
     */
    __device__ __forceinline__ void applyRuntimeCandidate(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        __shared__ std::uint64_t command_storage[
            sizeof(MoEOverlayDeviceControllerCommandHeader) /
            sizeof(std::uint64_t)];
        __shared__ std::uint64_t digest_parts[kControllerThreads];
        __shared__ std::uint64_t byte_parts[kControllerThreads];
        __shared__ std::uint32_t invalid_parts[kControllerThreads];
        __shared__ std::uint32_t missing_parts[kControllerThreads];
        __shared__ std::uint32_t runtime_invalid_parts[kControllerThreads];
        __shared__ std::uint32_t ready;
        __shared__ std::uint32_t candidate_bank;
        __shared__ std::uint32_t published_bank;

        auto &command = *reinterpret_cast<
            MoEOverlayDeviceControllerCommandHeader *>(command_storage);
        const auto &binding = launch.binding;
        const auto &publication = launch.runtime_publication;
        if (threadIdx.x == 0u)
        {
            ready = 0u;
            storeSystemRelease(
                &publication.apply_status->code,
                raw(MoEOverlayDeviceRuntimeApplyCode::Idle));

            const bool state_ready = waitForState(
                binding,
                MoEOverlayDeviceControllerState::PreparingFollowers);
            command = snapshotPeerRecord(binding.command);
            const std::uint64_t transaction = loadSystemAcquire(
                &binding.controller->transaction_id);
            const std::uint64_t selector = loadLocalEpochWord(
                &publication.epoch_control->published_selector);
            published_bank =
                static_cast<std::uint32_t>(selector & 1u);
            candidate_bank =
                published_bank < kDeviceMoEOverlayEpochBankCount
                    ? published_bank ^ 1u
                    : kDeviceMoEOverlayInvalidBank;

            const auto local_metadata = snapshotPeerRecord(
                binding.participants + binding.participant_id);
            const auto *local_group = groupLayout(
                binding, binding.group_id);
            const bool command_header_valid =
                state_ready && transaction != 0u &&
                command.magic == kMoEOverlayDeviceControllerMagic &&
                command.version == kMoEOverlayDeviceControllerVersion &&
                command.kind == raw(
                    MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement) &&
                command.transaction_id == transaction &&
                command.topology_fingerprint == binding.topology_fingerprint &&
                command.base_epoch ==
                    loadSystemAcquire(&binding.controller->base_epoch) &&
                command.candidate_epoch ==
                    loadSystemAcquire(&binding.controller->candidate_epoch) &&
                command.base_epoch < 0xffffffffULL &&
                command.candidate_epoch == command.base_epoch + 1u &&
                command.candidate_epoch <= 0xffffffffULL &&
                command.command_count != 0u &&
                command.command_count <=
                    loadPeerPublished(&binding.layout->command_capacity) &&
                command.command_count <= publication.arrival_capacity &&
                command.parallel_command_count == command.command_count &&
                command.movement_round_count == 1u &&
                command.hazard_count == 0u &&
                command.packed_weight_bytes != 0u &&
                loadSystemAcquire(
                    &binding.controller->command_transaction) == transaction &&
                publication.layer_count ==
                    loadPeerPublished(&binding.layout->num_layers) &&
                publication.expert_count ==
                    loadPeerPublished(&binding.layout->num_experts) &&
                participantMetadataValid(
                    binding, local_metadata, binding.participant_id) &&
                local_metadata.group_id == binding.group_id &&
                local_metadata.domain_participant_index ==
                    publication.domain_participant_id &&
                local_group != nullptr &&
                local_group->participant_count ==
                    publication.domain_participant_count;
            const bool epoch_control_valid =
                (selector >> 1u) != 0u &&
                candidate_bank < kDeviceMoEOverlayEpochBankCount &&
                published_bank < kDeviceMoEOverlayEpochBankCount &&
                candidate_bank != published_bank &&
                loadLocalEpochWord(
                    &publication.epoch_control
                         ->bank_epochs[published_bank]) ==
                    command.base_epoch &&
                loadLocalEpochWord(
                    &publication.epoch_control
                         ->bank_states[published_bank]) ==
                    raw(DeviceMoEOverlayEpochBankState::Published);

            if (!command_header_valid || !epoch_control_valid)
            {
                finishRuntimeApplyStatus(
                    publication.apply_status,
                    command_header_valid
                        ? MoEOverlayDeviceRuntimeApplyCode::InvalidReservation
                        : MoEOverlayDeviceRuntimeApplyCode::InvalidCommand,
                    transaction,
                    command.base_epoch,
                    command.candidate_epoch,
                    candidate_bank,
                    command.command_count,
                    0u,
                    0u,
                    0u,
                    0u);
                failParticipant(
                    binding,
                    command_header_valid
                        ? MoEOverlayDeviceControllerError::InvalidState
                        : MoEOverlayDeviceControllerError::InvalidCommand);
            }
            else if (!waitForRuntimeTransport(
                         binding,
                         transaction,
                         command.command_digest))
            {
                finishRuntimeApplyStatus(
                    publication.apply_status,
                    MoEOverlayDeviceRuntimeApplyCode::MissingArrival,
                    transaction,
                    command.base_epoch,
                    command.candidate_epoch,
                    candidate_bank,
                    command.command_count,
                    0u,
                    0u,
                    command.command_count,
                    0u);
            }
            else
            {
                ready = 1u;
            }
        }
        __syncthreads();
        if (ready == 0u)
            return;

        std::uint64_t digest = threadIdx.x == 0u
                                   ? moeOverlayCommandDigestSeed(
                                         command.command_count)
                                   : 0u;
        constexpr std::uint32_t kWordsPerCommand =
            sizeof(MoEOverlayDeviceMovementCommand) /
            sizeof(std::uint64_t);
        const auto *command_words = reinterpret_cast<const std::uint64_t *>(
            binding.command_entries);
        const std::uint64_t word_count =
            static_cast<std::uint64_t>(command.command_count) *
            kWordsPerCommand;
        for (std::uint64_t word = threadIdx.x;
             word < word_count;
             word += blockDim.x)
        {
            digest ^= moeOverlayCommandDigestWord(
                loadPeerPublished(command_words + word), word);
        }

        std::uint64_t packed_bytes = 0u;
        std::uint32_t invalid = 0u;
        std::uint32_t missing = 0u;
        for (std::uint32_t ordinal = threadIdx.x;
             ordinal < command.command_count;
             ordinal += blockDim.x)
        {
            const auto entry = snapshotPeerRecord(
                binding.command_entries + ordinal);
            packed_bytes = saturatingAdd(
                packed_bytes, entry.payload_bytes);

            bool entry_valid = moeOverlayMovementCommandValid(
                entry,
                ordinal,
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement,
                binding.layout->participant_count,
                publication.layer_count,
                publication.expert_count,
                command.base_epoch,
                command.candidate_epoch);
            if (entry_valid)
            {
                const auto source = snapshotPeerRecord(
                    binding.participants + entry.source_participant);
                const auto destination = snapshotPeerRecord(
                    binding.participants + entry.destination_participant);
                entry_valid = participantMetadataValid(
                                  binding,
                                  source,
                                  entry.source_participant) &&
                    participantMetadataValid(
                        binding,
                        destination,
                        entry.destination_participant);
            }
            if (ordinal != 0u)
            {
                const auto previous = snapshotPeerRecord(
                    binding.command_entries + ordinal - 1u);
                entry_valid = entry_valid &&
                    moeOverlayIndependentCanonicalDestination(
                        previous, entry);
            }
            if (entry_valid)
            {
                const auto *runtime = runtimeLayer(
                    publication.runtime_layers, entry.layer);
                entry_valid = runtime->banks[published_bank]
                                  .overlay_route_participant[entry.expert] ==
                    static_cast<std::int32_t>(entry.source_participant);
            }
            if (entry_valid &&
                entry.destination_participant == binding.participant_id &&
                !runtimeArrivalReady(
                    preparedArrival(publication, ordinal), entry.expert))
            {
                ++missing;
            }
            if (!entry_valid)
                ++invalid;
        }

        std::uint32_t runtime_invalid = 0u;
        for (std::uint32_t layer = threadIdx.x;
             layer < publication.layer_count;
             layer += blockDim.x)
        {
            const auto *runtime = runtimeLayer(
                publication.runtime_layers, layer);
            if (runtime->active_bank != published_bank ||
                runtime->active_epoch != command.base_epoch ||
                runtime->expert_count != publication.expert_count ||
                runtime->banks[published_bank].epoch != command.base_epoch ||
                runtime->banks[published_bank].expert_count !=
                    publication.expert_count ||
                runtimeParticipantId(
                    publication.runtime_layers, layer) !=
                    publication.domain_participant_id ||
                runtimeParticipantCount(
                    publication.runtime_layers, layer) !=
                    publication.domain_participant_count ||
                !runtimeOwnsPlacementBanks(
                    publication.runtime_layers, layer))
            {
                ++runtime_invalid;
            }
        }

        digest_parts[threadIdx.x] = digest;
        byte_parts[threadIdx.x] = packed_bytes;
        invalid_parts[threadIdx.x] = invalid;
        missing_parts[threadIdx.x] = missing;
        runtime_invalid_parts[threadIdx.x] = runtime_invalid;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2u;
             stride != 0u;
             stride >>= 1u)
        {
            if (threadIdx.x < stride)
            {
                digest_parts[threadIdx.x] ^=
                    digest_parts[threadIdx.x + stride];
                byte_parts[threadIdx.x] = saturatingAdd(
                    byte_parts[threadIdx.x], byte_parts[threadIdx.x + stride]);
                invalid_parts[threadIdx.x] += invalid_parts[threadIdx.x + stride];
                missing_parts[threadIdx.x] += missing_parts[threadIdx.x + stride];
                runtime_invalid_parts[threadIdx.x] +=
                    runtime_invalid_parts[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u)
        {
            if (digest_parts[0] != command.command_digest ||
                byte_parts[0] != command.packed_weight_bytes)
            {
                ++invalid_parts[0];
            }
            if (invalid_parts[0] != 0u || missing_parts[0] != 0u ||
                runtime_invalid_parts[0] != 0u)
            {
                const auto code = runtime_invalid_parts[0] != 0u
                                      ? MoEOverlayDeviceRuntimeApplyCode::
                                            InvalidRuntime
                                  : missing_parts[0] != 0u
                                      ? MoEOverlayDeviceRuntimeApplyCode::
                                            MissingArrival
                                      : MoEOverlayDeviceRuntimeApplyCode::
                                            InvalidCommand;
                finishRuntimeApplyStatus(
                    publication.apply_status,
                    code,
                    command.transaction_id,
                    command.base_epoch,
                    command.candidate_epoch,
                    candidate_bank,
                    command.command_count,
                    0u,
                    0u,
                    missing_parts[0],
                    runtime_invalid_parts[0]);
                failParticipant(
                    binding,
                    code == MoEOverlayDeviceRuntimeApplyCode::InvalidCommand
                        ? MoEOverlayDeviceControllerError::InvalidCommand
                        : MoEOverlayDeviceControllerError::InvalidState);
                ready = 0u;
            }
        }
        __syncthreads();
        if (ready == 0u)
            return;

        if (threadIdx.x == 0u &&
            !reserveLocalEpochCandidate(
                publication.epoch_control,
                command.candidate_epoch,
                publication.epoch_status,
                candidate_bank,
                published_bank))
        {
            finishRuntimeApplyStatus(
                publication.apply_status,
                MoEOverlayDeviceRuntimeApplyCode::InvalidReservation,
                command.transaction_id,
                command.base_epoch,
                command.candidate_epoch,
                candidate_bank,
                command.command_count,
                0u,
                0u,
                0u,
                0u);
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            ready = 0u;
        }
        __syncthreads();
        if (ready == 0u)
            return;

        // Clone first so every untouched expert and layer remains part of the
        // candidate. The command is not allowed to expose a partially updated
        // bank, and the local selector is still pinned to published_bank.
        for (std::uint32_t layer = 0u;
             layer < publication.layer_count;
             ++layer)
        {
            auto *runtime = runtimeLayer(publication.runtime_layers, layer);
            auto &candidate = runtime->banks[candidate_bank];
            const auto &published = runtime->banks[published_bank];
            for (std::uint32_t expert = threadIdx.x;
                 expert < publication.expert_count;
                 expert += blockDim.x)
            {
                candidate.experts[expert] = published.experts[expert];
                candidate.local_compute_mask[expert] =
                    published.local_compute_mask[expert];
                candidate.replica_role[expert] =
                    published.replica_role[expert];
                candidate.resident_participant_mask[expert] =
                    published.resident_participant_mask[expert];
                candidate.overlay_route_participant[expert] =
                    published.overlay_route_participant[expert];
            }
            __syncthreads();
            if (threadIdx.x == 0u)
            {
                candidate.expert_count = published.expert_count;
                candidate.multi_resident_expert_count =
                    published.multi_resident_expert_count;
                candidate.transient_placement_observed =
                    published.transient_placement_observed;
            }
            __syncthreads();
        }

        // Commands are conflict-free, so the complete movement wave can update
        // independent expert slots in parallel without atomics or serialization.
        for (std::uint32_t ordinal = threadIdx.x;
             ordinal < command.command_count;
             ordinal += blockDim.x)
        {
            const auto entry = snapshotPeerRecord(
                binding.command_entries + ordinal);
            const auto destination = snapshotPeerRecord(
                binding.participants + entry.destination_participant);
            auto *runtime = runtimeLayer(
                publication.runtime_layers, entry.layer);
            auto &candidate = runtime->banks[candidate_bank];
            auto &descriptor = candidate.experts[entry.expert];

            descriptor = RuntimeExpertDescriptorView{};
            descriptor.logical_expert_id =
                static_cast<std::int32_t>(entry.expert);
            // Zero-initialization would make slot zero look meaningful on a
            // non-owning participant. Keep every metadata-only or external
            // descriptor explicitly non-executable until the exact transfer
            // destination installs its prepared pointer-bearing triple.
            descriptor.owner_participant = -1;
            descriptor.local_slot = -1;
            candidate.overlay_route_participant[entry.expert] =
                static_cast<std::int32_t>(entry.destination_participant);
            candidate.local_compute_mask[entry.expert] = 0u;
            candidate.replica_role[entry.expert] = kRuntimeReplicaRoleNone;
            candidate.resident_participant_mask[entry.expert] = 0u;

            if (destination.group_id == binding.group_id)
            {
                const std::uint32_t local_destination =
                    destination.domain_participant_index;
                descriptor.owner_participant =
                    static_cast<std::int32_t>(local_destination);
                candidate.resident_participant_mask[entry.expert] =
                    1u << local_destination;
                if (entry.destination_participant == binding.participant_id)
                {
                    descriptor = preparedArrival(publication, ordinal);
                    descriptor.logical_expert_id =
                        static_cast<std::int32_t>(entry.expert);
                    descriptor.owner_participant =
                        static_cast<std::int32_t>(local_destination);
                    descriptor.flags =
                        (descriptor.flags & kRuntimeExpertFlagTransferSlot) |
                        kRuntimeExpertFlagValid |
                        kRuntimeExpertFlagResident |
                        kRuntimeExpertFlagPreferredOwner |
                        kRuntimeExpertFlagLocalCompute;
                    candidate.local_compute_mask[entry.expert] = 1u;
                    candidate.replica_role[entry.expert] =
                        kRuntimeReplicaRolePrimary;
                }
            }
            else
            {
                descriptor.owner_participant = -1;
                descriptor.local_slot = -1;
            }
        }
        __threadfence();
        __syncthreads();

        // Recompute the scalar instead of trying to patch it per command. This
        // keeps it exact even if the source bank contained durable replicas.
        for (std::uint32_t layer = 0u;
             layer < publication.layer_count;
             ++layer)
        {
            const auto *runtime = runtimeLayer(
                publication.runtime_layers, layer);
            const auto &candidate = runtime->banks[candidate_bank];
            std::uint32_t multi_resident = 0u;
            for (std::uint32_t expert = threadIdx.x;
                 expert < publication.expert_count;
                 expert += blockDim.x)
            {
                const std::uint32_t mask =
                    candidate.resident_participant_mask[expert];
                multi_resident +=
                    mask != 0u && (mask & (mask - 1u)) != 0u ? 1u : 0u;
            }
            invalid_parts[threadIdx.x] = multi_resident;
            __syncthreads();
            for (std::uint32_t stride = blockDim.x / 2u;
                 stride != 0u;
                 stride >>= 1u)
            {
                if (threadIdx.x < stride)
                {
                    invalid_parts[threadIdx.x] +=
                        invalid_parts[threadIdx.x + stride];
                }
                __syncthreads();
            }
            if (threadIdx.x == 0u)
            {
                auto *mutable_runtime = runtimeLayer(
                    publication.runtime_layers, layer);
                mutable_runtime->banks[candidate_bank]
                    .multi_resident_expert_count = invalid_parts[0];
                mutable_runtime->banks[candidate_bank].epoch =
                    static_cast<std::uint32_t>(command.candidate_epoch);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0u)
        {
            std::uint32_t changed_layers = 0u;
            for (std::uint32_t layer = 0u;
                 layer < publication.layer_count;
                 ++layer)
            {
                bool changed = false;
                for (std::uint32_t ordinal = 0u;
                     ordinal < command.command_count && !changed;
                     ++ordinal)
                {
                    changed = loadPeerPublished(
                                  &binding.command_entries[ordinal].layer) ==
                        layer;
                }
                changed_layers += changed ? 1u : 0u;
            }
            finishRuntimeApplyStatus(
                publication.apply_status,
                MoEOverlayDeviceRuntimeApplyCode::Success,
                command.transaction_id,
                command.base_epoch,
                command.candidate_epoch,
                candidate_bank,
                command.command_count,
                command.command_count,
                changed_layers,
                0u,
                0u);
            // This participant owns the exact runtime table just completed.
            // Publish only after all descriptor writes and the semantic status
            // are globally visible to its group-root maintenance kernel.
            __threadfence_system();
            storeSystemRelease(
                &binding.local_participant_record->prepared_transaction,
                command.transaction_id);
        }
    }

    /**
     * @brief Validate and publish one device-local candidate under sole authority.
     *
     * This action owns Candidate-to-Ready, the selector flip, and canonical
     * runtime activation. Keeping those mutations behind the live mapped
     * controller state prevents an unconditional later graph node from
     * publishing after a sibling participant has failed the transaction.
     * External topology-wide admission still names the base epoch during
     * participant fan-out, so inference retains the old local bank until every
     * participant has release-published this action.
     */
    __device__ __forceinline__ void publishRuntimeCandidate(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        __shared__ std::uint64_t command_storage[
            sizeof(MoEOverlayDeviceControllerCommandHeader) /
            sizeof(std::uint64_t)];
        __shared__ std::uint64_t apply_status_storage[
            sizeof(MoEOverlayDeviceRuntimeApplyStatus) /
            sizeof(std::uint64_t)];
        __shared__ std::uint32_t invalid_parts[kControllerThreads];
        __shared__ std::uint32_t ready;
        __shared__ std::uint32_t candidate_bank;
        __shared__ std::uint32_t prior_bank;

        auto &command = *reinterpret_cast<
            MoEOverlayDeviceControllerCommandHeader *>(command_storage);
        auto &apply_status = *reinterpret_cast<
            MoEOverlayDeviceRuntimeApplyStatus *>(apply_status_storage);
        const auto &binding = launch.binding;
        const auto &publication = launch.runtime_publication;

        if (threadIdx.x == 0u)
        {
            ready = 0u;
            const bool state_ready = waitForState(
                binding,
                MoEOverlayDeviceControllerState::PublishingFollowers);
            const std::uint64_t transaction = loadSystemAcquire(
                &binding.controller->transaction_id);
            command = snapshotPeerRecord(binding.command);
            apply_status = snapshotPeerRecord(publication.apply_status);
            candidate_bank = apply_status.candidate_bank;
            const std::uint64_t selector = loadLocalEpochWord(
                &publication.epoch_control->published_selector);
            prior_bank = static_cast<std::uint32_t>(selector & 1u);

            const bool valid = state_ready && transaction != 0u &&
                command.kind == raw(
                    MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement) &&
                command.transaction_id == transaction &&
                command.topology_fingerprint == binding.topology_fingerprint &&
                command.base_epoch ==
                    loadSystemAcquire(&binding.controller->base_epoch) &&
                command.candidate_epoch ==
                    loadSystemAcquire(&binding.controller->candidate_epoch) &&
                loadSystemAcquire(
                    &binding.controller->commit_transaction) == transaction &&
                loadSystemAcquire(
                    &binding.local_participant_record->prepared_transaction) ==
                    transaction &&
                apply_status.succeeded() &&
                apply_status.transaction_id == transaction &&
                apply_status.base_epoch == command.base_epoch &&
                apply_status.candidate_epoch == command.candidate_epoch &&
                apply_status.candidate_bank == candidate_bank &&
                loadLocalEpochWord(&publication.epoch_status->code) == raw(
                    DeviceMoEOverlayEpochStatusCode::Success) &&
                publication.epoch_status->operation == raw(
                    DeviceMoEOverlayEpochOperation::ReserveCandidate) &&
                publication.epoch_status->epoch == command.candidate_epoch &&
                publication.epoch_status->bank == candidate_bank &&
                candidate_bank < kDeviceMoEOverlayEpochBankCount &&
                prior_bank < kDeviceMoEOverlayEpochBankCount &&
                candidate_bank != prior_bank &&
                publication.epoch_status->selector == selector &&
                loadLocalEpochWord(
                    &publication.epoch_control->bank_epochs[candidate_bank]) ==
                    command.candidate_epoch &&
                loadLocalEpochWord(
                    &publication.epoch_control->bank_states[candidate_bank]) ==
                    raw(DeviceMoEOverlayEpochBankState::Candidate) &&
                loadLocalEpochWord(
                    &publication.epoch_control->bank_epochs[prior_bank]) ==
                    command.base_epoch &&
                loadLocalEpochWord(
                    &publication.epoch_control->bank_states[prior_bank]) ==
                    raw(DeviceMoEOverlayEpochBankState::Published);
            if (valid)
                ready = 1u;
            else
                failParticipant(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
        }
        __syncthreads();
        if (ready == 0u)
            return;

        std::uint32_t invalid = 0u;
        for (std::uint32_t layer = threadIdx.x;
             layer < publication.layer_count;
             layer += blockDim.x)
        {
            const auto *runtime = runtimeLayer(
                publication.runtime_layers, layer);
            if (runtime->active_bank != prior_bank ||
                runtime->active_epoch != command.base_epoch ||
                runtime->expert_count != publication.expert_count ||
                runtime->banks[prior_bank].epoch != command.base_epoch ||
                runtime->banks[candidate_bank].epoch !=
                    command.candidate_epoch ||
                runtime->banks[candidate_bank].expert_count !=
                    publication.expert_count ||
                !runtimeOwnsPlacementBanks(
                    publication.runtime_layers, layer))
            {
                ++invalid;
            }
        }
        invalid_parts[threadIdx.x] = invalid;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x / 2u;
             stride != 0u;
             stride >>= 1u)
        {
            if (threadIdx.x < stride)
            {
                invalid_parts[threadIdx.x] +=
                    invalid_parts[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0u && invalid_parts[0] != 0u)
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            ready = 0u;
        }
        __syncthreads();
        if (ready == 0u)
            return;

        if (threadIdx.x == 0u &&
            !publishLocalEpochCandidate(
                publication.epoch_control,
                command.candidate_epoch,
                publication.epoch_status,
                candidate_bank,
                prior_bank))
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            ready = 0u;
        }
        __syncthreads();
        if (ready == 0u)
            return;

        for (std::uint32_t layer = threadIdx.x;
             layer < publication.layer_count;
             layer += blockDim.x)
        {
            auto *runtime = runtimeLayer(publication.runtime_layers, layer);
            runtime->active_bank = candidate_bank;
            runtime->active_epoch =
                static_cast<std::uint32_t>(command.candidate_epoch);
        }
        __threadfence_system();
        if (threadIdx.x == 0u)
        {
            storeSystemRelease(
                &binding.local_participant_record->published_transaction,
                command.transaction_id);
        }
    }

    /**
     * @brief Reclaim and acknowledge the old local bank under sole authority.
     *
     * The scheduler submits this action only after every participant's separate
     * readiness receipt names the base epoch. It verifies local and topology
     * readiness once, empties the old bank, and publishes retirement. Any stale
     * receipt is fatal; this action never publishes Busy and is never replayed
     * as a reader-drain polling loop.
     */
    __device__ __forceinline__ void publishRuntimeRetirement(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const auto &publication = launch.runtime_publication;
        if (threadIdx.x != 0u)
            return;

        const bool state_ready = waitForState(
            binding,
            MoEOverlayDeviceControllerState::RetiringDurableEpoch);
        const std::uint64_t transaction = loadSystemAcquire(
            &binding.controller->transaction_id);
        const auto command = snapshotPeerRecord(binding.command);
        const auto prior_status = snapshotPeerRecord(
            publication.epoch_status);
        const std::uint64_t selector = loadLocalEpochWord(
            &publication.epoch_control->published_selector);
        const std::uint32_t active_bank =
            static_cast<std::uint32_t>(selector & 1u);
        const std::uint32_t retiring_bank = findLocalEpochBank(
            publication.epoch_control, command.base_epoch);
        const bool already_retired =
            loadSystemAcquire(
                &binding.local_participant_record->retired_epoch) ==
            command.base_epoch;
        if (already_retired)
        {
            // Exact duplicate submission is an idempotent terminal no-op; any
            // disagreement in the active generation remains fatal.
            const bool active_valid = state_ready && transaction != 0u &&
                command.transaction_id == transaction &&
                command.candidate_epoch ==
                    loadSystemAcquire(
                        &binding.controller->candidate_epoch) &&
                active_bank < kDeviceMoEOverlayEpochBankCount &&
                loadLocalEpochWord(
                    &publication.epoch_control->bank_epochs[active_bank]) ==
                    command.candidate_epoch &&
                loadLocalEpochWord(
                    &publication.epoch_control->bank_states[active_bank]) ==
                    raw(DeviceMoEOverlayEpochBankState::Published) &&
                findLocalEpochBank(
                    publication.epoch_control, command.base_epoch) ==
                    kDeviceMoEOverlayInvalidBank;
            if (!active_valid)
            {
                failParticipant(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
            }
            return;
        }
        const bool prior_publish =
            prior_status.code == raw(
                DeviceMoEOverlayEpochStatusCode::Success) &&
            prior_status.operation == raw(
                DeviceMoEOverlayEpochOperation::PublishCandidate) &&
            prior_status.epoch == command.candidate_epoch &&
            prior_status.bank == active_bank;
        bool valid = state_ready && transaction != 0u &&
            command.kind == raw(
                MoEOverlayDeviceControllerTransactionKind::
                    DynamicPlacement) &&
            command.transaction_id == transaction &&
            command.topology_fingerprint == binding.topology_fingerprint &&
            command.base_epoch ==
                loadSystemAcquire(&binding.controller->base_epoch) &&
            command.candidate_epoch ==
                loadSystemAcquire(&binding.controller->candidate_epoch) &&
            loadSystemAcquire(
                &binding.controller->admission_transaction) == transaction &&
            loadSystemAcquire(&binding.controller->admission_epoch) ==
                command.candidate_epoch &&
            loadSystemAcquire(
                &binding.local_participant_record->published_transaction) ==
                transaction &&
            prior_publish &&
            active_bank < kDeviceMoEOverlayEpochBankCount &&
            retiring_bank < kDeviceMoEOverlayEpochBankCount &&
            retiring_bank != active_bank &&
            loadLocalEpochWord(
                &publication.epoch_control->bank_epochs[retiring_bank]) ==
                command.base_epoch &&
            loadLocalEpochWord(
                &publication.epoch_control->bank_states[retiring_bank]) ==
                raw(DeviceMoEOverlayEpochBankState::Retiring) &&
            loadLocalEpochWord(
                &publication.epoch_control->bank_epochs[active_bank]) ==
                command.candidate_epoch &&
            loadLocalEpochWord(
                &publication.epoch_control->bank_states[active_bank]) ==
                raw(DeviceMoEOverlayEpochBankState::Published);
        for (std::uint32_t layer = 0u;
             valid && layer < publication.layer_count;
             ++layer)
        {
            const auto *runtime = runtimeLayer(
                publication.runtime_layers, layer);
            valid = runtime->active_bank == active_bank &&
                runtime->active_epoch == command.candidate_epoch &&
                runtime->banks[active_bank].epoch ==
                    command.candidate_epoch &&
                runtimeOwnsPlacementBanks(
                    publication.runtime_layers, layer);
        }
        if (!valid)
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            return;
        }

        /* The scheduler reached this graph only after it acquire-observed the
         * complete topology readiness set.  Do not probe the device-wide
         * acquisition guard again: post-receipt acquisitions target E+1 and
         * may overlap this reclamation without making E unsafe. */
        if (loadSystemAcquire(
                &binding.local_participant_record->retirement_ready_epoch) !=
            command.base_epoch)
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            return;
        }

        /* The scheduler submits this bounded retirement epoch only after every
         * topology participant's immutable readiness word is acquire-visible.
         * Re-check once here to make a stale scheduler ticket fatal without
         * ever leaving a polling kernel resident on a compute unit. */
        if (!allParticipantsRetirementReady(binding, command.base_epoch))
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            return;
        }

        if (!retireReadinessCertifiedLocalEpoch(
                publication.epoch_control,
                command.base_epoch,
                publication.epoch_status))
        {
            /* Readiness is monotonic after topology-wide admission: every
             * request that selected E already owns all participant guards.
             * Busy here is therefore a broken transaction-ordering proof. */
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        __threadfence_system();
        storeSystemRelease(
            &binding.local_participant_record->retired_epoch,
            command.base_epoch);
    }

    /**
     * @brief Keep a follower graph live until the leader publishes completion.
     *
     * A follower acknowledgement is only an input to the leader's final
     * transition. Without this acquire edge, the follower stream can record
     * its terminal event while the shared controller is still in
     * `PublishingFollowers`. The setup observer would then race the leader
     * even though every protocol write was correct. Dynamic and LLEP graphs
     * can reuse the same action after their respective final acknowledgement.
     */
    __device__ __forceinline__ void awaitTransactionComplete(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        (void)waitForState(
            binding, MoEOverlayDeviceControllerState::Complete);
    }

    /** Join the global commit edge before mutating this participant's selector. */
    __device__ __forceinline__ void awaitRuntimeCommit(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const bool state_ready = waitForState(
            binding,
            MoEOverlayDeviceControllerState::PublishingFollowers);
        const std::uint64_t transaction = loadSystemAcquire(
            &binding.controller->transaction_id);
        if (!state_ready || transaction == 0u ||
            loadSystemAcquire(&binding.controller->commit_transaction) !=
                transaction)
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidTransaction);
        }
    }

    /** Join global admission before attempting this device's reader drain. */
    __device__ __forceinline__ void awaitRuntimeRetirement(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const bool state_ready = waitForState(
            binding,
            MoEOverlayDeviceControllerState::RetiringDurableEpoch);
        if (!state_ready ||
            loadSystemAcquire(&binding.controller->transaction_kind) !=
                raw(MoEOverlayDeviceControllerTransactionKind::
                        DynamicPlacement))
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
        }
    }

    /**
     * @brief Publish this device's bounded old-bank grace-period receipt.
     *
     * The action never waits for readers. It is captured once after durable
     * publication and may also run directly after an inference release. The
     * last release therefore authors the immutable readiness ticket used to
     * submit the retained retirement epoch; the host cannot infer or fabricate
     * reader drainage.
     */
    __device__ __forceinline__ void publishRuntimeRetirementReadiness(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const auto &readiness = launch.retirement_readiness;
        /* This action is also retained in the ordinary inference-release
         * epilogue. Most releases do not overlap a movement transaction, so
         * retirement absence is the expected bounded no-op. Waiting here
         * would turn an otherwise complete inference graph into a resident
         * maintenance kernel. */
        const bool state_ready =
            loadSystemAcquire(&binding.controller->state) ==
            raw(MoEOverlayDeviceControllerState::RetiringDurableEpoch);
        if (!state_ready)
            return;
        const auto command = snapshotPeerRecord(binding.command);
        if (loadSystemAcquire(&binding.controller->transaction_kind) !=
                raw(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement) ||
            command.transaction_id !=
                loadSystemAcquire(&binding.controller->transaction_id) ||
            command.topology_fingerprint != binding.topology_fingerprint)
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        if (command.command_count == 0u ||
            command.candidate_epoch == command.base_epoch)
        {
            failParticipant(
                binding,
                MoEOverlayDeviceControllerError::InvalidCommand);
            return;
        }

        if (loadSystemAcquire(
                &binding.local_participant_record->retirement_ready_epoch) ==
            command.base_epoch)
        {
            return;
        }

        std::uint32_t retiring_bank = kDeviceMoEOverlayInvalidBank;
        if (!localEpochRetirementReady(
                readiness.epoch_control,
                command.base_epoch,
                &retiring_bank))
        {
            return;
        }
        __threadfence_system();
        storeSystemRelease(
            &binding.local_participant_record->retirement_ready_epoch,
            command.base_epoch);
    }

    /** Complete an authenticated zero-movement Dynamic decision immediately. */
    __device__ __forceinline__ void completeEmptyDynamicDecision(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.authorityLeader() || !waitForState(
                binding,
                MoEOverlayDeviceControllerState::PreparingFollowers))
        {
            if (binding.authorityLeader())
                failLeader(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        const std::uint64_t transaction = loadSystemAcquire(
            &binding.controller->transaction_id);
        const auto command = snapshotPeerRecord(binding.command);
        if (command.command_count != 0u)
            return;
        if (transaction == 0u ||
            command.kind != raw(
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement) ||
            command.transaction_id != transaction ||
            command.topology_fingerprint != binding.topology_fingerprint ||
            command.base_epoch != command.candidate_epoch ||
            command.base_epoch != loadSystemAcquire(
                &binding.controller->current_durable_epoch) ||
            command.packed_weight_bytes != 0u ||
            command.command_digest != moeOverlayCommandDigestSeed(0u))
        {
            failLeader(
                binding,
                MoEOverlayDeviceControllerError::InvalidCommand);
            return;
        }
        storeSystemRelease(
            &binding.controller->admission_epoch,
            command.base_epoch);
        storeSystemRelease(
            &binding.controller->admission_transaction,
            transaction);
        storeSystemRelease(
            &binding.controller->completed_transaction,
            transaction);
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::Complete));
    }

    /** Acknowledge group-local preparation of the immutable command. */
    __device__ __forceinline__ void acknowledgePrepared(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.groupRoot() || !waitForState(
                binding,
                MoEOverlayDeviceControllerState::PreparingFollowers))
        {
            return;
        }
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        const auto command = snapshotPeerRecord(binding.command);
        if (transaction == 0u || command.transaction_id != transaction ||
            command.topology_fingerprint != binding.topology_fingerprint ||
            command.parallel_command_count != command.command_count ||
            command.movement_round_count !=
                (command.command_count == 0u ? 0u : 1u) ||
            command.hazard_count != 0u ||
            loadSystemAcquire(&binding.controller->command_transaction) !=
                transaction)
        {
            failGroup(binding, MoEOverlayDeviceControllerError::InvalidCommand);
            return;
        }
        if (command.packed_weight_bytes != 0u &&
            !waitForLocalTransport(
                binding,
                TransportWordField::PreparedTransaction,
                transaction,
                transaction,
                command.command_digest))
        {
            return;
        }
        if (command.kind == raw(
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement) &&
            command.command_count != 0u &&
            !waitForAllGroupParticipants(
                binding,
                ParticipantWordField::PreparedTransaction,
                transaction))
        {
            return;
        }
        __threadfence_system();
        storeSystemRelease(
            &binding.local_group->prepared_transaction, transaction);
    }

    /** Begin the globally ordered publish only after all preparation acks. */
    __device__ __forceinline__ void beginCommit(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        if (!binding.authorityLeader() || transaction == 0u ||
            !waitForState(
                binding,
                MoEOverlayDeviceControllerState::PreparingFollowers) ||
            !waitForAllGroups(
                binding,
                GroupWordField::PreparedTransaction,
                transaction))
        {
            if (binding.authorityLeader())
                failLeader(binding, MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        storeSystemRelease(
            &binding.controller->commit_transaction, transaction);
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::PublishingFollowers));
    }

    /** Acknowledge local runtime-bank publication after the command commit. */
    __device__ __forceinline__ void acknowledgePublished(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.groupRoot() || !waitForState(
                binding,
                MoEOverlayDeviceControllerState::PublishingFollowers))
        {
            return;
        }
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        if (transaction == 0u ||
            loadSystemAcquire(&binding.controller->commit_transaction) !=
                transaction)
        {
            failGroup(binding, MoEOverlayDeviceControllerError::InvalidTransaction);
            return;
        }
        const auto command = snapshotPeerRecord(binding.command);
        if (command.transaction_id != transaction ||
            command.topology_fingerprint != binding.topology_fingerprint ||
            (command.packed_weight_bytes != 0u &&
             !waitForLocalTransport(
                 binding,
                 TransportWordField::PublishedTransaction,
                 transaction,
                 transaction,
                 command.command_digest)))
        {
            if (command.transaction_id != transaction ||
                command.topology_fingerprint !=
                    binding.topology_fingerprint)
            {
                failGroup(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidCommand);
            }
            return;
        }
        if (command.kind == raw(
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement) &&
            command.command_count != 0u &&
            !waitForAllGroupParticipants(
                binding,
                ParticipantWordField::PublishedTransaction,
                transaction))
        {
            return;
        }
        __threadfence_system();
        storeSystemRelease(
            &binding.local_group->published_transaction, transaction);
    }

    /** Linearize admission after every group published its local bank. */
    __device__ __forceinline__ void publishAdmission(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        if (!binding.authorityLeader() || transaction == 0u ||
            !waitForState(
                binding,
                MoEOverlayDeviceControllerState::PublishingFollowers) ||
            !waitForAllGroups(
                binding,
                GroupWordField::PublishedTransaction,
                transaction) ||
            loadSystemAcquire(&binding.controller->commit_transaction) !=
                transaction)
        {
            if (binding.authorityLeader())
                failLeader(binding, MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        const auto kind = static_cast<
            MoEOverlayDeviceControllerTransactionKind>(
            loadSystemAcquire(&binding.controller->transaction_kind));
        if (kind ==
            MoEOverlayDeviceControllerTransactionKind::DynamicPlacement)
        {
            /* Hysteresis changes only at the durable linearization point.
             * Proposal, preparation, and local-bank publication remain
             * reversible and therefore must never age an expert. Every
             * command was authenticated before commit and destinations are
             * disjoint, so these mapped words can be release-published in
             * canonical command order immediately before the epoch itself. */
            const std::uint32_t command_count = loadPeerPublished(
                &binding.command->command_count);
            const std::uint32_t expert_count = loadPeerPublished(
                &binding.layout->num_experts);
            for (std::uint32_t ordinal = 0u;
                 ordinal < command_count;
                 ++ordinal)
            {
                const auto command = snapshotPeerRecord(
                    binding.command_entries + ordinal);
                if (command.layer >= binding.layout->num_layers ||
                    command.expert >= expert_count)
                {
                    failLeader(
                        binding,
                        MoEOverlayDeviceControllerError::InvalidCommand);
                    return;
                }
                storeSystemRelease(
                    binding.economy_last_moved +
                        static_cast<std::uint64_t>(command.layer) *
                            expert_count +
                        command.expert,
                    transaction);
            }
            storeSystemRelease(
                &binding.controller->current_durable_epoch,
                binding.controller->candidate_epoch);
            storeSystemRelease(
                &binding.controller->admission_epoch,
                binding.controller->candidate_epoch);
        }
        else
        {
            storeSystemRelease(
                &binding.controller->admission_epoch,
                binding.controller->base_epoch);
        }
        if (kind ==
            MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP)
        {
            storeSystemRelease(
                &binding.controller->active_llep_transaction, transaction);
        }
        storeSystemRelease(
            &binding.controller->admission_transaction, transaction);
        if (kind ==
            MoEOverlayDeviceControllerTransactionKind::StaticCheck)
        {
            // This monotonic terminal receipt survives the next transaction's
            // state transition, unlike the transient Complete state itself.
            storeSystemRelease(
                &binding.controller->completed_transaction, transaction);
        }
        storeSystemRelease(
            &binding.controller->state,
            kind == MoEOverlayDeviceControllerTransactionKind::StaticCheck
                ? raw(MoEOverlayDeviceControllerState::Complete)
                : raw(MoEOverlayDeviceControllerState::Admitted));
    }

    /** Enter Dynamic grace-period retirement. */
    __device__ __forceinline__ void beginDynamicRetirement(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.authorityLeader() || !waitForState(
                binding, MoEOverlayDeviceControllerState::Admitted) ||
            loadSystemAcquire(&binding.controller->transaction_kind) !=
                raw(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement))
        {
            if (binding.authorityLeader())
                failLeader(binding, MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::RetiringDurableEpoch));
    }

    /** Publish one group's completion of old-bank reclamation. */
    __device__ __forceinline__ void acknowledgeRetired(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.groupRoot() || !waitForState(
                binding,
                MoEOverlayDeviceControllerState::RetiringDurableEpoch))
        {
            return;
        }
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        const std::uint64_t base_epoch =
            loadSystemAcquire(&binding.controller->base_epoch);
        const std::uint64_t candidate_epoch =
            loadSystemAcquire(&binding.controller->candidate_epoch);
        const auto command = snapshotPeerRecord(binding.command);
        if (command.transaction_id != transaction ||
            command.topology_fingerprint != binding.topology_fingerprint ||
            (candidate_epoch != base_epoch &&
             command.packed_weight_bytes != 0u &&
             !waitForLocalTransport(
                 binding,
                 TransportWordField::RetiredEpoch,
                 transaction,
                 base_epoch,
                 command.command_digest)))
        {
            if (command.transaction_id != transaction ||
                command.topology_fingerprint !=
                    binding.topology_fingerprint)
            {
                failGroup(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidCommand);
            }
            return;
        }
        if (candidate_epoch != base_epoch &&
            !waitForAllGroupParticipants(
                binding,
                ParticipantWordField::RetiredEpoch,
                base_epoch))
        {
            return;
        }
        __threadfence_system();
        storeSystemRelease(
            &binding.local_group->retired_epoch,
            base_epoch);
    }

    /** Complete Dynamic only after every prior bank is retired. */
    __device__ __forceinline__ void completeDynamicRetirement(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.authorityLeader() || !waitForState(
                binding,
                MoEOverlayDeviceControllerState::RetiringDurableEpoch) ||
            !waitForAllGroups(
                binding,
                GroupWordField::RetiredEpoch,
                loadSystemAcquire(&binding.controller->base_epoch)))
        {
            if (binding.authorityLeader())
                failLeader(binding, MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        storeSystemRelease(
            &binding.controller->completed_transaction,
            loadSystemAcquire(&binding.controller->transaction_id));
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::Complete));
    }

    /** Enter current-batch LLEP restoration after sparse return completion. */
    __device__ __forceinline__ void beginLLEPRestore(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        if (!binding.authorityLeader() || !waitForState(
                binding, MoEOverlayDeviceControllerState::Admitted) ||
            loadSystemAcquire(&binding.controller->transaction_kind) !=
                raw(MoEOverlayDeviceControllerTransactionKind::CurrentBatchLLEP) ||
            loadSystemAcquire(
                &binding.controller->active_llep_transaction) != transaction)
        {
            if (binding.authorityLeader())
                failLeader(binding, MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::RestoringLLEP));
    }

    /** Publish restoration of one group's durable owner-only assignment. */
    __device__ __forceinline__ void acknowledgeLLEPRestored(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        if (!binding.groupRoot() || !waitForState(
                binding,
                MoEOverlayDeviceControllerState::RestoringLLEP))
        {
            return;
        }
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        const auto command = snapshotPeerRecord(binding.command);
        if (command.transaction_id != transaction ||
            command.topology_fingerprint != binding.topology_fingerprint ||
            (command.packed_weight_bytes != 0u &&
             !waitForLocalTransport(
                 binding,
                 TransportWordField::RestoredTransaction,
                 transaction,
                 transaction,
                 command.command_digest)))
        {
            if (command.transaction_id != transaction ||
                command.topology_fingerprint !=
                    binding.topology_fingerprint)
            {
                failGroup(
                    binding,
                    MoEOverlayDeviceControllerError::InvalidCommand);
            }
            return;
        }
        __threadfence_system();
        storeSystemRelease(
            &binding.local_group->restored_transaction, transaction);
    }

    /** Complete LLEP only after every transient assignment is gone. */
    __device__ __forceinline__ void completeLLEPRestore(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const std::uint64_t transaction =
            loadSystemAcquire(&binding.controller->transaction_id);
        if (!binding.authorityLeader() || !waitForState(
                binding,
                MoEOverlayDeviceControllerState::RestoringLLEP) ||
            !waitForAllGroups(
                binding,
                GroupWordField::RestoredTransaction,
                transaction))
        {
            if (binding.authorityLeader())
                failLeader(binding, MoEOverlayDeviceControllerError::InvalidState);
            return;
        }
        storeSystemRelease(
            &binding.controller->active_llep_transaction,
            std::uint64_t{0u});
        storeSystemRelease(
            &binding.controller->completed_transaction, transaction);
        storeSystemRelease(
            &binding.controller->state,
            raw(MoEOverlayDeviceControllerState::Complete));
    }

    /** @return Lifecycle state in which one Dynamic runtime action is legal. */
    __device__ __forceinline__ MoEOverlayDeviceControllerState
    dynamicRuntimeActionState(
        MoEOverlayDeviceControllerAction action) noexcept
    {
        if (action ==
            MoEOverlayDeviceControllerAction::ApplyRuntimeCandidate)
        {
            return MoEOverlayDeviceControllerState::PreparingFollowers;
        }
        if (action ==
            MoEOverlayDeviceControllerAction::PublishRuntimeCandidate)
        {
            return MoEOverlayDeviceControllerState::PublishingFollowers;
        }
        return MoEOverlayDeviceControllerState::RetiringDurableEpoch;
    }

    /**
     * @brief Validate the empty-command branch of a complete Dynamic graph.
     *
     * A retained transaction cannot omit runtime nodes merely because policy
     * may decide not to move an expert: the decision is device-authored after
     * graph launch.  Once the action's exact lifecycle state is acquire-visible,
     * an authenticated empty command makes apply, publish, and retirement true
     * no-ops.  This keeps zero movement on the same captured transaction as
     * physical movement without manufacturing an RCU bank or durable epoch.
     */
    __device__ __forceinline__ bool validatedDynamicRuntimeNoOp(
        const MoEOverlayDeviceControllerActionLaunch &launch) noexcept
    {
        const auto &binding = launch.binding;
        const std::uint64_t transaction = loadSystemAcquire(
            &binding.controller->transaction_id);
        const auto command = snapshotPeerRecord(binding.command);
        const bool valid = transaction != 0u &&
            loadSystemAcquire(&binding.controller->state) ==
                raw(dynamicRuntimeActionState(launch.action)) &&
            loadSystemAcquire(&binding.controller->transaction_kind) ==
                raw(MoEOverlayDeviceControllerTransactionKind::DynamicPlacement) &&
            loadSystemAcquire(&binding.controller->command_transaction) ==
                transaction &&
            command.kind == raw(
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement) &&
            command.transaction_id == transaction &&
            command.topology_fingerprint == binding.topology_fingerprint &&
            command.base_epoch ==
                loadSystemAcquire(&binding.controller->base_epoch) &&
            command.candidate_epoch == command.base_epoch &&
            command.candidate_epoch ==
                loadSystemAcquire(&binding.controller->candidate_epoch) &&
            command.command_count == 0u &&
            command.parallel_command_count == 0u &&
            command.movement_round_count == 0u &&
            command.hazard_count == 0u &&
            command.packed_weight_bytes == 0u &&
            command.command_digest == moeOverlayCommandDigestSeed(0u);
        if (!valid)
        {
            failParticipant(
                binding, MoEOverlayDeviceControllerError::InvalidCommand);
        }
        return valid;
    }

    /** Execute exactly one typed transition on a mapped device-owned timeline. */
    static __global__ void controllerActionKernel(
        MoEOverlayDeviceControllerActionLaunch launch)
    {
        if (blockIdx.x != 0u)
            return;
        __shared__ std::uint32_t launch_valid;
        __shared__ std::uint32_t runtime_noop;
        __shared__ DynamicPolicyScratch dynamic_policy_scratch;
        if (threadIdx.x == 0u)
        {
            launch_valid =
                launch.valid() && validIdentity(launch.binding) ? 1u : 0u;
        }
        __syncthreads();
        if (launch_valid == 0u)
        {
            if (threadIdx.x == 0u)
            {
                if (launch.binding.authorityLeader())
                    failLeader(
                        launch.binding,
                        MoEOverlayDeviceControllerError::InvalidControl);
                else
                    failParticipant(
                        launch.binding,
                        MoEOverlayDeviceControllerError::InvalidControl);
            }
            return;
        }

        if (launch.action ==
            MoEOverlayDeviceControllerAction::PublishParticipantSnapshot)
        {
            publishParticipantSnapshot(launch);
            return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::PublishGroupSnapshot)
        {
            publishGroupSnapshot(launch);
            return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::PublishCommand)
        {
            publishCommand(launch);
            return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::AuthorStaticPolicy)
        {
            if (threadIdx.x == 0u)
                authorStaticPolicy(launch);
            return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::AuthorDynamicPolicy)
        {
            if (threadIdx.x == 0u)
                authorDynamicPolicy(launch, dynamic_policy_scratch);
            return;
        }

        const bool runtime_action =
            launch.action ==
                MoEOverlayDeviceControllerAction::ApplyRuntimeCandidate ||
            launch.action ==
                MoEOverlayDeviceControllerAction::PublishRuntimeCandidate ||
            launch.action ==
                MoEOverlayDeviceControllerAction::PublishRuntimeRetirement;
        if (runtime_action)
        {
            if (threadIdx.x == 0u)
            {
                runtime_noop = 0u;
                if (!waitForState(
                        launch.binding,
                        dynamicRuntimeActionState(launch.action)))
                {
                    failParticipant(
                        launch.binding,
                        MoEOverlayDeviceControllerError::InvalidState);
                    runtime_noop = 2u;
                }
                else if (loadPeerPublished(
                             &launch.binding.command->command_count) == 0u)
                {
                    runtime_noop =
                        validatedDynamicRuntimeNoOp(launch) ? 1u : 2u;
                }
            }
            __syncthreads();
            // Both a valid no-op and a validation failure terminate this node.
            // A non-empty command continues into the full runtime operation.
            if (runtime_noop != 0u)
                return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::ApplyRuntimeCandidate)
        {
            applyRuntimeCandidate(launch);
            return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::PublishRuntimeCandidate)
        {
            publishRuntimeCandidate(launch);
            return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::PublishRuntimeRetirement)
        {
            publishRuntimeRetirement(launch);
            return;
        }
        if (launch.action ==
            MoEOverlayDeviceControllerAction::
                PublishRuntimeRetirementReadiness)
        {
            if (threadIdx.x == 0u)
                publishRuntimeRetirementReadiness(launch);
            return;
        }
        if (threadIdx.x != 0u)
            return;

        switch (launch.action)
        {
        case MoEOverlayDeviceControllerAction::BeginTransaction:
            beginTransaction(launch);
            break;
        case MoEOverlayDeviceControllerAction::PublishCommand:
            break; // Handled above by the complete validation block.
        case MoEOverlayDeviceControllerAction::AcknowledgePrepared:
            acknowledgePrepared(launch);
            break;
        case MoEOverlayDeviceControllerAction::BeginCommit:
            beginCommit(launch);
            break;
        case MoEOverlayDeviceControllerAction::AcknowledgePublished:
            acknowledgePublished(launch);
            break;
        case MoEOverlayDeviceControllerAction::PublishAdmission:
            publishAdmission(launch);
            break;
        case MoEOverlayDeviceControllerAction::BeginDynamicRetirement:
            beginDynamicRetirement(launch);
            break;
        case MoEOverlayDeviceControllerAction::AcknowledgeRetired:
            acknowledgeRetired(launch);
            break;
        case MoEOverlayDeviceControllerAction::CompleteDynamicRetirement:
            completeDynamicRetirement(launch);
            break;
        case MoEOverlayDeviceControllerAction::CompleteEmptyDynamicDecision:
            completeEmptyDynamicDecision(launch);
            break;
        case MoEOverlayDeviceControllerAction::BeginLLEPRestore:
            beginLLEPRestore(launch);
            break;
        case MoEOverlayDeviceControllerAction::AcknowledgeLLEPRestored:
            acknowledgeLLEPRestored(launch);
            break;
        case MoEOverlayDeviceControllerAction::CompleteLLEPRestore:
            completeLLEPRestore(launch);
            break;
        case MoEOverlayDeviceControllerAction::AuthorStaticPolicy:
            break; // Handled above by the single-writer policy block.
        case MoEOverlayDeviceControllerAction::AuthorDynamicPolicy:
            break; // Handled above by the deterministic policy block.
        case MoEOverlayDeviceControllerAction::ApplyRuntimeCandidate:
            break; // Handled above by the complete all-layer apply block.
        case MoEOverlayDeviceControllerAction::PublishRuntimeCandidate:
            break; // Handled above by the complete RCU publication block.
        case MoEOverlayDeviceControllerAction::PublishRuntimeRetirement:
            break; // Handled above by the reader-drain acknowledgement block.
        case MoEOverlayDeviceControllerAction::
            PublishRuntimeRetirementReadiness:
            break; // Handled above by the bounded grace-period receipt block.
        case MoEOverlayDeviceControllerAction::AwaitTransactionComplete:
            awaitTransactionComplete(launch);
            break;
        case MoEOverlayDeviceControllerAction::AwaitRuntimeCommit:
            awaitRuntimeCommit(launch);
            break;
        case MoEOverlayDeviceControllerAction::AwaitRuntimeRetirement:
            awaitRuntimeRetirement(launch);
            break;
        case MoEOverlayDeviceControllerAction::Invalid:
        case MoEOverlayDeviceControllerAction::PublishParticipantSnapshot:
        case MoEOverlayDeviceControllerAction::PublishGroupSnapshot:
            failLeader(
                launch.binding,
                MoEOverlayDeviceControllerError::InvalidControl);
            break;
        }
    }
} // namespace llaminar2::moe_overlay_controller_device
