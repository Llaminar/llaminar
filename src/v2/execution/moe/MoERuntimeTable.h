/**
 * @file MoERuntimeTable.h
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#pragma once

#include "DeviceMoEOverlayEpochABI.h"
#include "DeviceMoEOverlayServiceTelemetry.h"
#include "DeviceMoERuntimeABI.h"
#include "LeastLoadedExpertAssignment.h"
#include "MoEOverlayActivationPacketABI.h"
#include "RuntimeExpertHistogramDrain.h"

#include "../../backends/DeviceId.h"
#include "../../kernels/common/DeviceMoEFloatingMatrixDesc.h"
#include "../../tensors/TensorKernels.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace llaminar2
{

    class DecodeExpertHistogram;
    class DeviceMoEOverlayEpochArena;
    using DeviceMoERuntimeHistogramBank =
        moe_runtime_abi::DeviceMoERuntimeHistogramBank;
    using RuntimeExpertHistogramSourceMask =
        std::array<bool, moe_runtime_abi::kHistogramSourceCount>;

    /** Every runtime phase is collected by a non-overlay table. */
    inline constexpr RuntimeExpertHistogramSourceMask
        kAllRuntimeExpertHistogramSources{true, true, true};

    inline constexpr uint32_t kDeviceMoEMaxExperts = 256;
    /**
     * @brief Largest transfer-directory slot representable by the runtime ABI.
     *
     * Logical expert IDs index fixed per-layer arrays and remain bounded by
     * @ref kDeviceMoEMaxExperts. Transfer slots instead name physical payload
     * allocations shared across every layer. Their descriptor field is signed
     * so negative values can remain invalid sentinels; positive slot IDs are
     * therefore total through INT32_MAX and are constrained in practice by the
     * directory's VRAM preflight, not by the number of experts in one layer.
     */
    inline constexpr uint32_t kDeviceMoEMaxTransferSlots = 0x7fffffffu;
    inline constexpr uint32_t kDeviceMoEMaxTopK = 16;
    inline constexpr uint32_t kDeviceMoEMaxParticipants = 8;

    enum class DeviceMoEExpertFlags : uint32_t
    {
        None = 0,
        Valid = 1u << 0,
        Resident = 1u << 1,
        Replicated = 1u << 2,
        PreferredOwner = 1u << 3,
        LocalCompute = 1u << 4,
        TransferSlot = 1u << 5,
    };

    constexpr DeviceMoEExpertFlags operator|(DeviceMoEExpertFlags lhs, DeviceMoEExpertFlags rhs) noexcept
    {
        return static_cast<DeviceMoEExpertFlags>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
    }

    constexpr DeviceMoEExpertFlags operator&(DeviceMoEExpertFlags lhs, DeviceMoEExpertFlags rhs) noexcept
    {
        return static_cast<DeviceMoEExpertFlags>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
    }

    constexpr DeviceMoEExpertFlags &operator|=(DeviceMoEExpertFlags &lhs, DeviceMoEExpertFlags rhs) noexcept
    {
        lhs = lhs | rhs;
        return lhs;
    }

    constexpr uint32_t toMoEExpertFlags(DeviceMoEExpertFlags flags) noexcept
    {
        return static_cast<uint32_t>(flags);
    }

    constexpr bool hasMoEExpertFlag(uint32_t flags, DeviceMoEExpertFlags flag) noexcept
    {
        return (flags & static_cast<uint32_t>(flag)) != 0;
    }

    enum class DeviceMoEReplicaRole : uint8_t
    {
        None = 0,
        Primary = 1,
        Replica = 2,
        PreferredReplica = 3,
    };

    struct DeviceMoEExpertDescriptor
    {
        /** Prepared NativeVNNI views used when weight_format is NativeVNNI. */
        DeviceNativeVNNIMatrixDesc gate;
        DeviceNativeVNNIMatrixDesc up;
        DeviceNativeVNNIMatrixDesc down;
        /** Contiguous floating views used for FP16, BF16, or FP32 experts. */
        DeviceMoEFloatingMatrixDesc floating_gate;
        DeviceMoEFloatingMatrixDesc floating_up;
        DeviceMoEFloatingMatrixDesc floating_down;
        int32_t logical_expert_id = -1;
        int32_t owner_participant = -1;
        int32_t local_slot = -1;
        uint32_t flags = 0;
        DeviceMoEWeightFormat weight_format = DeviceMoEWeightFormat::NativeVNNI;
        uint32_t reserved = 0;

        /** @return Whether the selected arithmetic family has a complete triple. */
        [[nodiscard]] constexpr bool weightsReady() const noexcept
        {
            if (weight_format == DeviceMoEWeightFormat::NativeVNNI)
                return gate.valid() && up.valid() && down.valid();
            return deviceMoEWeightFormatIsFloating(weight_format) &&
                   floating_gate.valid() &&
                   floating_up.valid() &&
                   floating_down.valid();
        }
    };

    struct DeviceMoEPlacementBank
    {
        DeviceMoEExpertDescriptor experts[kDeviceMoEMaxExperts] = {};
        uint8_t local_compute_mask[kDeviceMoEMaxExperts] = {};
        uint8_t replica_role[kDeviceMoEMaxExperts] = {};
        uint32_t resident_participant_mask[kDeviceMoEMaxExperts] = {};
        /**
         * Overlay-wide logical packet destination for each expert.
         *
         * This identity is deliberately distinct from
         * DeviceMoEExpertDescriptor::owner_participant, which is local to one
         * homogeneous compute domain. Values are stable logical participant
         * IDs from the global owner map; `-1` means that no overlay endpoint is
         * assigned. Publishing this array in the same double-buffered bank as
         * residency prevents a captured sparse dispatcher from observing a
         * new owner with an old routing target.
         */
        int32_t overlay_route_participant[kDeviceMoEMaxExperts] = {};
        uint32_t epoch = 0;
        uint32_t expert_count = 0;
        /// Number of experts resident on more than one participant.
        /// Decode uses this as a cheap hot-cache router-stats gate.
        uint32_t multi_resident_expert_count = 0;
        /**
         * @brief Domain-wide marker for request-lifetime transfer-slot placement.
         *
         * This marker is published from the globally agreed transfer plan on
         * every participant. It must never be inferred from a local expert
         * descriptor because only the transfer destination owns that
         * descriptor. Prefix capture uses the marker to preserve exact logical
         * placement symmetrically while excluding pointer-bearing slot
         * descriptors; restore rebuilds those payloads from immutable owners
         * in a dedicated captured device transaction.
         */
        uint32_t transient_placement_observed = 0;
    };

    struct DeviceMoELayerRuntime
    {
        uint32_t active_bank = 0;
        uint32_t active_epoch = 0;
        uint32_t expert_count = 0;
        uint32_t top_k = 0;
        DeviceMoEPlacementBank banks[2] = {};

        int32_t topk_expert_ids[kDeviceMoEMaxTopK] = {};
        float topk_weights[kDeviceMoEMaxTopK] = {};
        /// Serial decode selected/local demand for this request generation.
        uint64_t decode_histogram[kDeviceMoEMaxExperts] = {};
        uint64_t decode_local_histogram[kDeviceMoEMaxExperts] = {};
        /// Real (unpadded) ordinary prefill selected/local demand.
        uint64_t prefill_histogram[kDeviceMoEMaxExperts] = {};
        uint64_t prefill_local_histogram[kDeviceMoEMaxExperts] = {};
        /// Accepted grouped-MTP verifier selected/local demand.
        uint64_t grouped_verifier_histogram[kDeviceMoEMaxExperts] = {};
        uint64_t grouped_verifier_local_histogram[kDeviceMoEMaxExperts] = {};
        uint64_t router_hot_cache_eligible_dispatches = 0;
        uint64_t router_hot_cache_used_dispatches = 0;
        uint64_t router_hot_cache_improved_dispatches = 0;
        uint64_t router_hot_cache_default_load_spread_total = 0;
        uint64_t router_hot_cache_actual_load_spread_total = 0;
        uint64_t router_hot_cache_load_spread_improvement_total = 0;
        uint64_t router_hot_cache_active_dispatches = 0;
        uint64_t router_hot_cache_miss_dispatches = 0;
        uint64_t router_hot_cache_selected_expert_slots = 0;
        uint64_t router_hot_cache_replicated_selected_expert_slots = 0;

        int32_t *route_expert_ids = nullptr;
        float *route_weights = nullptr;
        int32_t *route_participant_ids = nullptr;
        /**
         * Per-layer immutable-address ledger for deferred verifier publication.
         *
         * Ordinary route scratch is intentionally shared by all serially
         * executed MoE layers on one device. The main MTP verifier, however,
         * publishes accepted routing history only after every layer and draft
         * sidecar has run. These two arrays retain the final expert and
         * participant assignment for this specific layer until that later
         * device-owned publication transaction consumes it.
         */
        int32_t *deferred_verifier_route_expert_ids = nullptr;
        int32_t *deferred_verifier_route_participant_ids = nullptr;
        int32_t *expert_counts = nullptr;
        int32_t *expert_offsets = nullptr;
        // Runtime grouped prefill stores original route-slot ids here
        // (`token * top_k + rank`), not bare token ids. This lets LLEP rewrite
        // route_participant_ids exactly without searching ambiguous top-k rows;
        // gather/scatter kernels derive the token row with `slot / top_k`.
        int32_t *grouped_token_ids = nullptr;
        float *grouped_route_weights = nullptr;
        float *grouped_gate_scratch = nullptr;
        float *grouped_up_scratch = nullptr;
        float *grouped_output_partials = nullptr;
        void *decode_scratch = nullptr;
        // reserved_ptrs[0]: resident-prefill LLEP split-end table,
        // [expert][participant] int32 cumulative route counts.
        // reserved_ptrs[1]: full current-batch LLEP assignment spans.
        // reserved_ptrs[2]: full current-batch LLEP expert-weight transfers.
        void *reserved_ptrs[3] = {};
        // reserved_u64[0]: current-batch LLEP assignment span capacity.
        // reserved_u64[1]: current-batch LLEP transfer capacity.
        // reserved_u64[2]: latest current-batch LLEP assignment span count.
        // reserved_u64[3]: latest current-batch LLEP transfer count.
        uint64_t reserved_u64[4] = {};
        uint32_t prefill_token_capacity = 0;
        uint32_t prefill_route_capacity = 0;
        uint32_t deferred_verifier_route_capacity = 0;
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
        /**
         * @brief Sticky proof that current-batch LLEP applied a payload move.
         *
         * This request/runtime-lifetime marker is deliberately separate from
         * DeviceMoEPlacementBank::transient_placement_observed. The bank marker
         * is portable prefix state and may therefore be restored from RAM or
         * disk before the current request plans any work. Only a transfer plan
         * explicitly tagged as current-batch LLEP may set this field, which
         * lets PerfStats prove that the production planner/materializer/apply
         * path actually ran instead of mistaking prefix rehydration for fresh
         * movement.
         */
        uint32_t current_batch_llep_movement_observed = 0;
        /**
         * @brief Sticky proof that current-batch LLEP consumed redistributed rows.
         *
         * The production current-batch assignment kernel sets this marker only
         * after validating the transfer/apply publication and observing a
         * non-empty assignment span whose destination differs from the expert's
         * authoritative owner. It therefore covers the economical resident
         * replica case where routed work moves but no expert payload needs to.
         *
         * Like @ref current_batch_llep_movement_observed, this is request-local
         * device state. It is never restored from prefix-cache state and reaches
         * PerfStats only through the existing stream-ordered maintenance status
         * publication.
         */
        uint32_t current_batch_llep_non_owner_assignment_observed = 0;
        /**
         * @brief Select this child table's request-local placement bank.
         *
         * An ExpertOverlay LLEP child normally reads the canonical parent bank
         * selected by @ref overlay_epoch_ticket. A successful current-batch
         * payload apply first clones that exact parent bank into one embedded
         * child bank, applies transient arrivals there, and publishes this bit
         * last. Subsequent captured kernels then consume the child bank while
         * still validating the pinned durable parent epoch. Request reset
         * clears the bit by restoring the immutable child template.
         */
        uint32_t current_batch_llep_transient_bank_active = 0;
        /**
         * Model-lifetime double-buffered routing evidence. Null selects the
         * embedded request-local arrays retained for CPU/legacy controllers.
         */
        DeviceMoERuntimeHistogramBank *runtime_histogram_banks = nullptr;
        /** Shared device scalar selecting the writable external bank. */
        const uint32_t *runtime_histogram_active_bank = nullptr;
        /**
         * @brief Stable request ticket selecting the immutable execution bank.
         *
         * A non-null pointer is the explicit ExpertOverlay mode contract.  Its
         * selected bank remains pinned across captured main-model, sparse
         * collective, continuation, and MTP graphs even while maintenance
         * publishes a newer epoch.  Null is reserved for non-overlay tables,
         * whose execution bank remains @ref active_bank.
         */
        const DeviceMoEOverlayEpochTicket *overlay_epoch_ticket = nullptr;
        /**
         * @brief Optional canonical placement banks for durable overlay roles.
         *
         * Main-model runtime layers leave this null and own their embedded
         * banks. MTP sidecars and current-batch LLEP children point it at the
         * corresponding main-model layer, so route scratch and histograms
         * remain child-local while every captured reader begins from one
         * device-owned durable placement authority. LLEP switches to an
         * embedded child bank only after a successful transient apply.
         */
        const DeviceMoEPlacementBank *overlay_placement_banks = nullptr;
        /**
         * @brief Device-owned result of the most recent epoch boundary operation.
         *
         * Production kernels consult this pointer only while reporting an already
         * fatal ticket contract. Keeping it beside the stable ticket lets CUDA and
         * ROCm attribute a zeroed ticket to its actual acquire/release boundary
         * without a host copy, synchronization, or shadow lifecycle.
         */
        const DeviceMoEOverlayEpochStatus *overlay_epoch_status = nullptr;
    };

    static_assert(std::is_trivially_copyable_v<DeviceMoEExpertDescriptor>);
    static_assert(
        sizeof(DeviceMoEExpertDescriptor) == 240,
        "host expert descriptors must retain the CUDA/ROCm runtime ABI size");
    static_assert(std::is_trivially_copyable_v<DeviceMoEPlacementBank>);
    static_assert(
        sizeof(DeviceMoEPlacementBank) ==
            moe_runtime_abi::kPlacementBankBytes,
        "host placement banks must retain the CUDA/ROCm runtime ABI size");
    static_assert(
        offsetof(DeviceMoEPlacementBank, overlay_route_participant) ==
            moe_runtime_abi::kOverlayRouteParticipantOffset,
        "host overlay route targets must retain the CUDA/ROCm ABI offset");
    static_assert(std::is_trivially_copyable_v<DeviceMoELayerRuntime>);
    static_assert(sizeof(DeviceMoELayerRuntime) ==
                  moe_runtime_abi::kLayerRuntimeBytes);
    static_assert(offsetof(DeviceMoELayerRuntime, route_participant_ids) ==
                  moe_runtime_abi::kRouteParticipantIdsOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime,
                 deferred_verifier_route_expert_ids) ==
        moe_runtime_abi::kDeferredVerifierExpertIdsOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime,
                 deferred_verifier_route_participant_ids) ==
        moe_runtime_abi::kDeferredVerifierParticipantIdsOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, expert_counts) ==
                  moe_runtime_abi::kExpertCountsOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, decode_local_histogram) ==
                  moe_runtime_abi::kDecodeLocalHistogramOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, prefill_local_histogram) ==
                  moe_runtime_abi::kPrefillLocalHistogramOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime, grouped_verifier_local_histogram) ==
        moe_runtime_abi::kGroupedVerifierLocalHistogramOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime, deferred_verifier_route_capacity) ==
        moe_runtime_abi::kDeferredVerifierRouteCapacityOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, participant_id) ==
                  moe_runtime_abi::kParticipantIdOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, participant_count) ==
                  moe_runtime_abi::kParticipantCountOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime,
                 current_batch_llep_movement_observed) ==
        moe_runtime_abi::kCurrentBatchLLEPMovementObservedOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime,
                 current_batch_llep_non_owner_assignment_observed) ==
        moe_runtime_abi::kCurrentBatchLLEPNonOwnerAssignmentObservedOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime,
                 current_batch_llep_transient_bank_active) ==
        moe_runtime_abi::kCurrentBatchLLEPTransientBankActiveOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, runtime_histogram_banks) ==
                  moe_runtime_abi::kRuntimeHistogramBanksOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime, runtime_histogram_active_bank) ==
        moe_runtime_abi::kRuntimeHistogramActiveBankOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, overlay_epoch_ticket) ==
                  moe_runtime_abi::kOverlayEpochTicketOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, overlay_placement_banks) ==
                  moe_runtime_abi::kOverlayPlacementBanksOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, overlay_epoch_status) ==
                  moe_runtime_abi::kOverlayEpochStatusOffset);

    /**
     * @brief Count active experts backed by graph-owned transient transfer slots.
     *
     * A transfer-slot flag alone only proves that a descriptor was materialized.
     * The active local-compute mask and the descriptor's valid, resident, and
     * local-compute flags prove that the descriptor was subsequently published
     * into the active placement bank and is eligible for expert execution.
     * This stronger predicate is therefore suitable for request-boundary
     * production-path evidence: a nonzero result means prefill movement reached
     * the applied runtime state, not merely that a planner proposed movement.
     *
     * @param state Participant-local runtime placement for one MoE layer.
     * @return Number of active, locally executable transfer-slot experts.
     */
    inline uint32_t deviceMoELayerActiveTransferSlotExpertCount(
        const DeviceMoELayerRuntime &state) noexcept
    {
        if (state.active_bank > 1u ||
            state.active_epoch == 0u ||
            state.expert_count > kDeviceMoEMaxExperts)
        {
            return 0u;
        }

        const auto &bank = state.banks[state.active_bank];
        const uint32_t expert_count =
            std::min<uint32_t>(state.expert_count, kDeviceMoEMaxExperts);
        constexpr uint32_t kAppliedTransferSlotFlags =
            toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
            toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
            toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute) |
            toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
        uint32_t count = 0u;
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            const uint32_t flags = bank.experts[expert].flags;
            if (bank.local_compute_mask[expert] != 0u &&
                (flags & kAppliedTransferSlotFlags) == kAppliedTransferSlotFlags)
            {
                ++count;
            }
        }
        return count;
    }

    /**
     * @brief Return whether the active placement consumes any transient payload.
     *
     * This convenience predicate deliberately delegates to the stricter count
     * helper so prefix-cache portability and PerfStats evidence share exactly
     * one definition of an applied transient expert.
     */
    inline bool deviceMoELayerUsesTransientLocalPayload(
        const DeviceMoELayerRuntime &state) noexcept
    {
        return deviceMoELayerActiveTransferSlotExpertCount(state) != 0u;
    }

    struct MoEPlacementUpdate
    {
        uint32_t epoch = 0;
        uint32_t expert_count = 0;
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
        std::vector<DeviceMoEExpertDescriptor> experts;
        std::vector<uint8_t> local_compute_mask;
        std::vector<uint8_t> replica_role;
        /// Optional [expert] bitmask of participants with resident weights.
        /// If omitted, prepareInactiveBank synthesizes owner/local residency.
        std::vector<uint32_t> resident_participant_mask;
        /**
         * Optional overlay-wide logical sparse-packet target per expert.
         *
         * Omission preserves non-overlay callers by deriving each value from
         * the descriptor's domain-local owner. ExpertOverlay construction must
         * provide the global owner-map values explicitly whenever participant
         * numbering differs across domains.
         */
        std::vector<int32_t> overlay_route_participant;
        /**
         * @brief Whether this bank includes request-lifetime transfer placement.
         *
         * Callers constructing the same logical bank for several participants
         * must publish the same value on every participant. Runtime transfer
         * apply derives it from the globally gathered transfer plan.
         */
        bool transient_placement_observed = false;
    };

    /**
     * @brief Exact host recipe and stable GPU destinations for one inactive bank.
     *
     * The host pointers describe setup-owned publication recipes, not a mirror
     * of mutable device execution state.  A maintenance publisher copies only
     * @ref host_bank into @ref device_bank, then publishes the two scalar
     * selector fields at the start of @ref device_runtime.  Copying the whole
     * runtime record would overwrite live routing scratch and histograms.
     */
    struct DeviceMoERuntimeBankPublicationRecipe
    {
        uint32_t bank = 0;
        uint32_t epoch = 0;
        const DeviceMoEPlacementBank *host_bank = nullptr;
        DeviceMoEPlacementBank *device_bank = nullptr;
        DeviceMoELayerRuntime *device_runtime = nullptr;

        /** @return Whether all identities name one publishable GPU bank. */
        [[nodiscard]] constexpr bool valid() const noexcept
        {
            return bank < kDeviceMoEOverlayEpochBankCount && epoch != 0u &&
                   host_bank != nullptr && device_bank != nullptr &&
                   device_runtime != nullptr && host_bank->epoch == epoch;
        }
    };

    /**
     * @brief Declare that every expert in a placement update is resident and
     * compute-ready on every participant in one replicated domain.
     *
     * A replicated graph still has a real multi-device topology. Treating each
     * participant as an independent one-device domain loses the owner map and
     * makes the same runtime table incompatible with apportioned prefill,
     * graph-side rebalance, and portable prefix restoration. This operation
     * publishes the complete topology atomically: participant identity, domain
     * width, canonical owner, all-participant residency, local-compute mask,
     * replica role, and descriptor flags.
     *
     * Callers must populate @p update.experts with one complete descriptor per
     * logical expert before invoking this function. Invalid participant IDs,
     * incomplete owner maps, and malformed descriptors throw immediately; no
     * partially declared replicated bank is permitted.
     *
     * @param update Placement update whose expert descriptors are finalized.
     * @param local_participant Participant represented by this device-local
     * runtime table.
     * @param participant_count Number of participants in the replicated domain.
     * @param owner_participants Canonical owner participant for every logical
     * expert, indexed by expert ID.
     */
    void declareFullyReplicatedPlacementTopology(
        MoEPlacementUpdate &update,
        int local_participant,
        int participant_count,
        const std::vector<int> &owner_participants);

    struct DeviceMoEPortableExpertRuntimeState
    {
        int32_t logical_expert_id = -1;
        int32_t owner_participant = -1;
        /**
         * Model-lifetime local slot, or -1 for a rolling transfer replica.
         *
         * Transfer-directory subscripts are graph-lifetime allocator state,
         * not portable prefix identity. A cached replica is reconstructed from
         * logical residency by a destination-local device lease and therefore
         * never serializes its prior rolling slot here.
         */
        int32_t local_slot = -1;
        uint32_t flags = 0;
        uint8_t local_compute = 0;
        uint8_t replica_role = static_cast<uint8_t>(DeviceMoEReplicaRole::None);
        uint32_t resident_participant_mask = 0;
    };

    struct DeviceMoEPortableLayerRuntimeState
    {
        uint32_t active_epoch = 0;
        uint32_t expert_count = 0;
        uint32_t top_k = 0;
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
        /**
         * @brief Whether logical restore requires rebuilding request-owned payloads.
         *
         * The portable record below contains logical placement only. When this
         * flag is set, one or more resident-participant bits name rolling
         * transfer-slot replicas whose pointer-bearing descriptors and rolling
         * slot indices deliberately are not serialized. Restore first publishes
         * immutable model placement plus a device transfer plan; a dedicated
         * captured graph then recreates those payloads before any restored-prefix
         * route is assigned.
         */
        uint32_t requires_device_payload_rehydration = 0;
        std::vector<DeviceMoEPortableExpertRuntimeState> experts;
        /// Serial-decode selected/local demand.
        std::vector<uint64_t> selected_histogram;
        std::vector<uint64_t> local_histogram;
        /// Ordinary-prefill selected/local demand.
        std::vector<uint64_t> prefill_selected_histogram;
        std::vector<uint64_t> prefill_local_histogram;
        /// Accepted grouped-verifier selected/local demand.
        std::vector<uint64_t> grouped_verifier_selected_histogram;
        std::vector<uint64_t> grouped_verifier_local_histogram;
    };

    /**
     * @brief Semantic placement effect produced by a portable runtime restore.
     *
     * Histograms and runtime epochs are request history, not expert placement.
     * A restore reports Changed only when ownership, participant residency,
     * local-compute eligibility, replica role, or graph-facing placement flags
     * differ from the live table's immutable model baseline.
     */
    enum class DeviceMoEPortablePlacementEffect : uint8_t
    {
        Unchanged,
        Changed,
    };

    /**
     * @brief Complete outcome of restoring pointer-free MoE runtime state.
     *
     * The result keeps archive validity, semantic placement mutation, and
     * pending device payload work separate. Callers must not infer movement
     * merely because histogram or epoch state was imported successfully.
     */
    struct DeviceMoEPortableRuntimeRestoreResult
    {
        bool restored = false;
        DeviceMoEPortablePlacementEffect placement_effect =
            DeviceMoEPortablePlacementEffect::Unchanged;
        bool requires_device_payload_rehydration = false;

        explicit constexpr operator bool() const noexcept { return restored; }
    };

    /**
     * @brief Device pointers backing one serial MoE route-planning transaction.
     *
     * These buffers contain only transient route/grouping data. They do not
     * contain graph-specific placement banks, epochs, or histograms. A graph
     * builder may therefore bind several non-concurrent runtime tables to the
     * same allocation while preserving independent placement metadata.
     */
    struct DeviceMoEPrefillRouteScratchBindings
    {
        int32_t *route_expert_ids = nullptr;
        float *route_weights = nullptr;
        int32_t *route_participant_ids = nullptr;
        int32_t *expert_counts = nullptr;
        int32_t *expert_offsets = nullptr;
        int32_t *grouped_token_ids = nullptr;
        float *grouped_route_weights = nullptr;
        int32_t *llep_split_ends = nullptr;
        least_loaded_ep::LeastLoadedExpertAssignmentSpan *llep_assignment_spans = nullptr;
        least_loaded_ep::LeastLoadedExpertWeightTransfer *llep_weight_transfers = nullptr;
        uint32_t token_capacity = 0;
        uint32_t route_capacity = 0;
        uint32_t expert_capacity = 0;
        uint32_t llep_plan_capacity = 0;
    };

    /**
     * @brief Immutable per-device scratch arena shared by serial GPU graphs.
     *
     * CUDA and HIP graph executables capture every pointer in @ref bindings_.
     * The arena consequently allocates its maximum capacity exactly once and
     * never exposes a resize operation. Main-model prefill, grouped verifier,
     * and MTP sidecar runtime tables may share it only when their execution is
     * ordered by one stream or an explicit producer/consumer event handoff.
     *
     * Placement tables remain graph-specific. Sharing this arena therefore
     * saves transient VRAM without allowing request resets or placement epochs
     * from one graph role to mutate another role's metadata.
     */
    class DeviceMoESerialRouteScratchArena final
    {
    public:
        struct Config
        {
            DeviceId device_id = DeviceId::cpu();
            int num_experts = 0;
            int top_k = 0;
            int token_capacity = 0;
        };

        explicit DeviceMoESerialRouteScratchArena(Config config);
        ~DeviceMoESerialRouteScratchArena();

        DeviceMoESerialRouteScratchArena(
            const DeviceMoESerialRouteScratchArena &) = delete;
        DeviceMoESerialRouteScratchArena &operator=(
            const DeviceMoESerialRouteScratchArena &) = delete;
        DeviceMoESerialRouteScratchArena(
            DeviceMoESerialRouteScratchArena &&) = delete;
        DeviceMoESerialRouteScratchArena &operator=(
            DeviceMoESerialRouteScratchArena &&) = delete;

        const DeviceId &deviceId() const noexcept { return device_id_; }
        int expertCount() const noexcept { return num_experts_; }
        int topK() const noexcept { return top_k_; }
        int tokenCapacity() const noexcept
        {
            return static_cast<int>(bindings_.token_capacity);
        }
        /**
         * @brief Return the requested device bytes owned by this arena.
         *
         * This is the sum of the ten immutable scratch allocations. It excludes
         * allocator bookkeeping and alignment padding, making it stable enough
         * for PerfStats and VRAM bill-of-materials regression checks.
         */
        size_t allocationBytes() const noexcept;

    private:
        friend class DeviceMoERuntimeTable;

        DeviceId device_id_;
        int num_experts_ = 0;
        int top_k_ = 0;
        DeviceMoEPrefillRouteScratchBindings bindings_;
    };

    class IMoERuntimeTable
    {
    public:
        virtual ~IMoERuntimeTable() = default;

        virtual DeviceMoELayerRuntime *deviceLayerState(int layer_idx) = 0;
        virtual int layerCount() const = 0;
        virtual const DeviceMoELayerRuntime &hostLayerState(int layer_idx) const = 0;
        /**
         * @brief Report whether the layer has no currently published decode bank.
         *
         * The host layer record is a publication recipe, not a coherence mirror
         * of a GPU table.  A device-owned reset can therefore invalidate the
         * published bank without rewriting that host recipe.  Every caller that
         * considers reusing a bank must reject it while this flag is set.
         */
        virtual bool decodeRuntimePublicationRequired(int layer_idx) const = 0;
        virtual bool prepareInactiveBank(int layer_idx, const MoEPlacementUpdate &update) = 0;
        virtual bool flipActiveBank(int layer_idx, uint32_t epoch, void *stream) = 0;
        virtual bool hasPrefillRouteScratchCapacity(int layer_idx, int token_count) const = 0;
        /**
         * @brief Prepare one exact stream to write device runtime histograms.
         *
         * GPU tables allocate the stream's reusable arrival event and enqueue
         * the one-time dependency on histogram-bank initialization here.  A
         * graph owner must call this method before `beginCapture()`; captured
         * stage execution may then record the producer identity without
         * importing uncaptured setup work into the graph.
         *
         * CPU tables and tables without asynchronous draining retain the stream
         * identity but require no backend resources.
         *
         * @param stream Exact non-null GPU producer stream, or the CPU producer
         *        identity for a host-owned table.
         */
        virtual void prepareDecodeHistogramProducerStream(void *stream) = 0;
        /**
         * @brief Publish the exact stream that just enqueued histogram writes.
         *
         * When asynchronous GPU draining is enabled, @p stream must already
         * have been admitted by @ref prepareDecodeHistogramProducerStream.
         * This method deliberately performs no allocation, copy, event wait, or
         * other backend operation, so it is safe to call while graph capture is
         * active.
         *
         * @param stream Exact non-null producer stream for mirrored GPU tables.
         */
        virtual void recordDecodeHistogramProducerStream(void *stream) = 0;
        /**
         * @brief Retire every graph-borrowed histogram producer stream.
         *
         * The graph owner calls this terminal transition after inference
         * submission has stopped but before destroying the contexts that own
         * producer streams. Implementations join the exact producer DAG onto
         * their maintenance authority and take one terminal fence. After this
         * method returns, later resource destruction must not dereference a
         * borrowed stream identity.
         *
         * CPU tables perform only the typed lifecycle transition. Repeated
         * calls are idempotent; producer admission or histogram progress after
         * retirement is invalid.
         */
        virtual void retireRuntimeHistogramProducerStreams() = 0;
        virtual void *decodeHistogramProducerStream() const = 0;
        /**
         * @brief Return the model-lifetime accepted-verifier publication stream.
         *
         * A mirrored table whose asynchronous source mask includes grouped
         * verification creates and admits this stream while installing its
         * histogram banks.  Accepted-state graph families borrow this exact
         * identity, so their first lazy materialization cannot introduce a new
         * producer after maintenance has sealed the topology.  The table owns
         * the stream; callers must neither replace nor destroy it.
         *
         * @return Exact non-null CUDA/HIP stream for grouped-verifier
         *         publication, or nullptr when this table owns no such source.
         */
        virtual void *groupedVerifierHistogramPublicationStream() const = 0;
        virtual bool syncDecodeHistogramToHost(DecodeExpertHistogram &histogram,
                                               void *stream = nullptr,
                                               bool reset_runtime_counts = true) = 0;
        /**
         * @brief Install persistent double-buffered device histogram storage.
         *
         * This is model-setup work and must run before any graph captures a
         * layer runtime pointer. @p sources selects the phases this table owns;
         * unselected phases may still be counted for diagnostics but are not
         * merged into the shared placement window.
         */
        virtual void enableAsyncDecodeHistogramDrain(
            RuntimeExpertHistogramSourceMask sources) = 0;
        /**
         * @brief Begin or poll one exact non-blocking runtime drain generation.
         *
         * Ready means the inactive bank was copied, cleared, and merged exactly
         * once. Pending must return without synchronizing a stream or device.
         */
        virtual RuntimeExpertHistogramDrainResult
        progressAsyncDecodeHistogramDrain(
            DecodeExpertHistogram &histogram) = 0;
        /**
         * @brief Publish a request-boundary route-admission phase.
         *
         * Mirrored tables join every exact producer into their one persistent
         * maintenance stream, publish one immutable writer state there, then
         * make every producer wait on that exact publication event. The method
         * never waits for a stream or device on the host.
         * CPU tables accept the phase without device work because their shared
         * @ref DecodeExpertHistogram is the direct authority.
         */
        virtual bool publishAsyncDecodeHistogramAdmission(
            RuntimeExpertHistogramAdmission admission) = 0;
        virtual bool captureDecodeHistogramCounts(std::vector<uint64_t> &selected_counts,
                                                  std::vector<uint64_t> &local_counts,
                                                  void *stream = nullptr) = 0;
        virtual bool restoreDecodeHistogramCounts(const uint64_t *selected_counts,
                                                  const uint64_t *local_counts,
                                                  size_t layer_count,
                                                  size_t expert_count,
                                                  void *stream = nullptr) = 0;
        virtual void resetDecodeHistogramCounts(void *stream = nullptr) = 0;
        virtual void resetDecodeRuntimeState(void *stream = nullptr) = 0;
        /**
         * @brief Return the first device-local ExpertOverlay service cell.
         *
         * A null result means this table was deliberately constructed without
         * Dynamic economy telemetry.  Child MTP/LLEP tables return their
         * canonical main table's allocation so every graph family contributes
         * to one participant/layer/phase measurement authority.
         */
        virtual DeviceMoEOverlayServiceTelemetryCell *
        deviceOverlayServiceTelemetry() const noexcept
        {
            return nullptr;
        }
        /**
         * @brief Return the stable timing cursor for one serial graph layer.
         * @param layer_idx Exact runtime layer index.
         * @return Device pointer, or null when telemetry is disabled/invalid.
         */
        virtual DeviceMoEOverlayServiceTelemetrySample *
        deviceOverlayServiceTelemetrySample(int layer_idx) const noexcept
        {
            (void)layer_idx;
            return nullptr;
        }
        /**
         * @brief Resolve the complete observation-only binding for one layer.
         *
         * Disabled telemetry returns an empty binding. A partial allocation is
         * a model-lifetime invariant violation and throws instead of allowing a
         * graph to silently omit evidence. This narrow view lets sparse child
         * executors report service time without borrowing runtime placement
         * authority from the table itself.
         *
         * @param layer_idx Exact model layer in this table.
         * @return Empty or complete device-local telemetry capability.
         * @throws std::out_of_range for an invalid layer.
         * @throws std::logic_error for a partial telemetry allocation.
         */
        DeviceMoEOverlayServiceTelemetryBinding
        deviceOverlayServiceTelemetryBinding(int layer_idx)
        {
            if (layer_idx < 0 || layer_idx >= layerCount())
            {
                throw std::out_of_range(
                    "MoE service telemetry binding layer is outside the runtime table");
            }

            auto *const telemetry = deviceOverlayServiceTelemetry();
            auto *const sample =
                deviceOverlayServiceTelemetrySample(layer_idx);
            if (!telemetry && !sample)
                return {};
            if (!telemetry || !sample)
            {
                throw std::logic_error(
                    "MoE service telemetry allocation is partial");
            }

            DeviceMoEOverlayServiceTelemetryBinding binding{
                .runtime_layer = deviceLayerState(layer_idx),
                .layer_telemetry =
                    telemetry + static_cast<std::size_t>(layer_idx) *
                                    kDeviceMoEOverlayServicePhaseCount,
                .sample = sample,
            };
            if (!binding.valid())
            {
                throw std::logic_error(
                    "MoE service telemetry allocation is partial");
            }
            return binding;
        }
        /**
         * @brief Resolve one exact captured ExpertOverlay packet-placement input.
         *
         * Non-overlay implementations return an invalid binding. Implementations
         * that expose it must bind both durable banks, their shared request
         * ticket, and the expert geometry from one placement authority.
         */
        virtual MoEOverlayRoutePlacementDeviceBinding
        overlayRoutePlacementBinding(int layer_idx) const
        {
            (void)layer_idx;
            return {};
        }
    };

    class DeviceMoERuntimeTable final : public IMoERuntimeTable
    {
    public:
        struct Config
        {
            DeviceId device_id = DeviceId::cpu();
            int num_layers = 0;
            int num_experts = 0;
            int top_k = 0;
            bool mirror_to_device = false;
            /**
             * @brief Allocate device-local service totals for Dynamic policy.
             *
             * This is valid only for a mirrored GPU table. Static overlays
             * leave it false and therefore pay no timing-kernel or storage
             * cost. A child table with @ref overlay_placement_source inherits
             * the canonical source allocation and must use the same value.
             */
            bool collect_overlay_service_telemetry = false;
            int prefill_token_capacity = 0;
            /**
             * @brief Rows retained independently for deferred MTP publication.
             *
             * A positive value allocates one compact expert/participant route
             * ledger per layer. It is a model-setup allocation with immutable
             * addresses, never a hot-path workspace or host mirror.
             */
            int deferred_verifier_token_capacity = 0;
            /**
             * @brief Optional immutable scratch owned by a serial graph domain.
             *
             * When present, every layer in this table binds the same stable
             * addresses and @ref ensurePrefillRouteScratchCapacity becomes a
             * validation-only operation. Exceeding the arena capacity is a
             * fatal planning error; captured addresses are never replaced.
             */
            std::shared_ptr<DeviceMoESerialRouteScratchArena>
                serial_route_scratch_arena;
            /**
             * @brief Optional shared request-lifetime ExpertOverlay epoch arena.
             *
             * Main-model and MTP runtime tables for one serial request family
             * bind the same arena and slot.  This is model topology, not
             * request data: reset and prefix restore preserve the pointer.
             */
            std::shared_ptr<DeviceMoEOverlayEpochArena> overlay_epoch_arena;
            /** @brief Immutable admission slot bound into every layer. */
            uint32_t overlay_epoch_ticket_slot = 0u;
            /**
             * @brief Optional canonical main-model placement authority.
             *
             * MTP sidecars and request-local LLEP child tables set this pointer.
             * The source must be the canonical mirrored main table on the same
             * device, cover every target layer, and consume the exact same
             * epoch ticket. Child tables may publish transient routing and
             * arrival state, but always resolve durable owners from these banks.
             * The graph builder owns both tables for their captured lifetime.
             */
            DeviceMoERuntimeTable *overlay_placement_source = nullptr;
        };

        explicit DeviceMoERuntimeTable(Config config);
        DeviceMoERuntimeTable(DeviceId device_id,
                              int num_layers,
                              int num_experts,
                              int top_k,
                              bool mirror_to_device = false);
        ~DeviceMoERuntimeTable() override;

        DeviceMoERuntimeTable(const DeviceMoERuntimeTable &) = delete;
        DeviceMoERuntimeTable &operator=(const DeviceMoERuntimeTable &) = delete;
        DeviceMoERuntimeTable(DeviceMoERuntimeTable &&) = delete;
        DeviceMoERuntimeTable &operator=(DeviceMoERuntimeTable &&) = delete;

        DeviceMoELayerRuntime *deviceLayerState(int layer_idx) override;
        bool prepareInactiveBank(int layer_idx, const MoEPlacementUpdate &update) override;
        bool flipActiveBank(int layer_idx, uint32_t epoch, void *stream) override;

        DeviceMoELayerRuntime &hostLayerState(int layer_idx);
        const DeviceMoELayerRuntime &hostLayerState(int layer_idx) const override;
        bool decodeRuntimePublicationRequired(int layer_idx) const override;
        bool hasPrefillRouteScratchCapacity(int layer_idx, int token_count) const override;
        /** @copydoc IMoERuntimeTable::prepareDecodeHistogramProducerStream */
        void prepareDecodeHistogramProducerStream(void *stream) override;
        /** @copydoc IMoERuntimeTable::recordDecodeHistogramProducerStream */
        void recordDecodeHistogramProducerStream(void *stream) override;
        /** @copydoc IMoERuntimeTable::retireRuntimeHistogramProducerStreams */
        void retireRuntimeHistogramProducerStreams() override;
        void *decodeHistogramProducerStream() const override;
        /** @copydoc IMoERuntimeTable::groupedVerifierHistogramPublicationStream */
        void *groupedVerifierHistogramPublicationStream() const override;
        bool syncDecodeHistogramToHost(DecodeExpertHistogram &histogram,
                                       void *stream = nullptr,
                                       bool reset_runtime_counts = true) override;
        /** @copydoc IMoERuntimeTable::enableAsyncDecodeHistogramDrain */
        void enableAsyncDecodeHistogramDrain(
            RuntimeExpertHistogramSourceMask sources) override;
        /** @copydoc IMoERuntimeTable::progressAsyncDecodeHistogramDrain */
        RuntimeExpertHistogramDrainResult
        progressAsyncDecodeHistogramDrain(
            DecodeExpertHistogram &histogram) override;
        /** @copydoc IMoERuntimeTable::publishAsyncDecodeHistogramAdmission */
        bool publishAsyncDecodeHistogramAdmission(
            RuntimeExpertHistogramAdmission admission) override;
        bool captureDecodeHistogramCounts(std::vector<uint64_t> &selected_counts,
                                          std::vector<uint64_t> &local_counts,
                                          void *stream = nullptr) override;
        bool restoreDecodeHistogramCounts(const uint64_t *selected_counts,
                                          const uint64_t *local_counts,
                                          size_t layer_count,
                                          size_t expert_count,
                                          void *stream = nullptr) override;
        void resetDecodeHistogramCounts(void *stream = nullptr) override;
        /**
         * @brief Restore an empty request-local placement using one ordered D2D copy.
         *
         * Mirrored GPU tables require an explicit stream. Their empty template
         * is allocated and populated during model setup, so request reset does
         * not upload stale host state, allocate a stream, or synchronize.
         */
        void resetDecodeRuntimeState(void *stream = nullptr) override;
        bool hasInitialRuntimeState() const noexcept;
        /**
         * @brief Return whether every retained layer owns a published baseline.
         *
         * @ref hasInitialRuntimeState answers whether any layer captured a
         * reset template, which remains useful for partial graph construction.
         * A topology-wide controller needs this stronger model-family
         * invariant: dormant MTP/NextN layers count exactly like active main
         * layers and may not remain at epoch zero.
         */
        [[nodiscard]] bool hasCompleteInitialRuntimeState() const noexcept;
        /**
         * @brief Seal and publish the complete setup-time runtime family.
         *
         * Graph construction may prepare a retained layer without ever
         * launching the graph that first references it. This one-shot typed
         * transition freezes the final complete host recipe as the immutable
         * baseline, uploads it to its device template, then publishes that
         * template to the live table on @p stream. Earlier per-layer captures
         * are construction checkpoints and may not substitute for this final
         * family snapshot. The stream's owner must publish the consumer event;
         * this method never synchronizes or allocates.
         *
         * @param stream Exact non-null controller publication stream.
         * @throws std::invalid_argument for a CPU table or null stream.
         * @throws std::logic_error when any retained layer lacks a baseline or
         *         the runtime family has already been published.
         * @throws std::runtime_error when a backend copy cannot be enqueued.
         */
        void sealAndPublishCompleteInitialRuntimeState(void *stream);
        /**
         * @brief Restore immutable model placement using one ordered D2D copy.
         *
         * Mirrored GPU tables require an explicit stream. The baseline keeps
         * model-lifetime scratch pointers and placement descriptors but clears
         * all request-local routing and histogram fields.
         */
        void restoreInitialRuntimeState(void *stream = nullptr);
        void syncRuntimeStateToHost(void *stream = nullptr);
        void restoreRuntimeStateSnapshot(const DeviceMoELayerRuntime *layers,
                                         size_t layer_count,
                                         void *stream = nullptr);
        /**
         * @brief Resolver used when a portable restore must rebind local expert payloads.
         *
         * Portable prefix-cache state stores logical placement and local-compute
         * intent, but deliberately does not embed pointer-bearing
         * DeviceMoELayerRuntime banks. A runtime that owns model-lifetime local
         * payloads may provide this resolver to turn a saved static
         * `(layer, expert, local_slot)` claim back into a live descriptor.
         *
         * Rolling transfer-slot indices are serialized as `-1` and are never
         * resolver candidates.
         * Portable capture preserves their pointer-free logical residency and
         * restore publishes a device-side rehydration plan from immutable owner
         * weights. Owned or statically mirrored local experts may still bind
         * from an existing ready runtime-bank descriptor because their payload
         * lifetime is the model itself.
         */
        using LocalPayloadDescriptorResolver =
            std::function<bool(int layer_idx,
                               int expert,
                               int local_slot,
                               DeviceMoEExpertDescriptor &out)>;

        bool capturePortableRuntimeState(std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
                                         void *stream = nullptr);
        DeviceMoEPortableRuntimeRestoreResult restorePortableRuntimeState(
            const std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
            void *stream = nullptr,
            const LocalPayloadDescriptorResolver &local_payload_resolver = {});
        void ensurePrefillRouteScratchCapacity(int token_capacity, void *stream = nullptr);

        const DeviceId &deviceId() const noexcept { return device_id_; }
        int layerCount() const noexcept override { return num_layers_; }
        int expertCount() const noexcept { return num_experts_; }
        int topK() const noexcept { return top_k_; }
        bool isMirroredToDevice() const noexcept { return mirror_to_device_; }
        bool usesImmutableSerialRouteScratch() const noexcept
        {
            return serial_route_scratch_arena_ != nullptr;
        }
        /** @return Whether this table executes through an epoch-pinned bank. */
        bool usesOverlayEpochTicket() const noexcept
        {
            return overlay_epoch_arena_ != nullptr;
        }
        /** @return Model-lifetime epoch arena identity, or null when unbound. */
        const DeviceMoEOverlayEpochArena *overlayEpochArena() const noexcept
        {
            return overlay_epoch_arena_.get();
        }
        /**
         * @return Stable backend-resident request ticket, or null for a
         * non-overlay table.
         */
        const DeviceMoEOverlayEpochTicket *overlayEpochTicket() const noexcept;
        /**
         * @return Status paired with @ref overlayEpochTicket, or null when
         *         this table is not bound to an overlay epoch arena.
         */
        const DeviceMoEOverlayEpochStatus *overlayEpochStatus() const noexcept;
        /** @return Mutable canonical main table, or null when this table owns placement. */
        DeviceMoERuntimeTable *overlayPlacementSource() noexcept
        {
            return overlay_placement_source_;
        }
        /** @return Canonical main table, or null when this table owns placement. */
        const DeviceMoERuntimeTable *overlayPlacementSource() const noexcept
        {
            return overlay_placement_source_;
        }
        /**
         * @brief Return the stable device address of one layer's placement banks.
         * @param layer_idx Model layer in this table.
         * @return First of the two embedded placement banks on host/device.
         */
        const DeviceMoEPlacementBank *devicePlacementBanks(
            int layer_idx) const;
        /**
         * @brief Resolve the prepared inactive bank without publishing it.
         * @param layer_idx Model layer whose inactive recipe is required.
         * @param epoch Exact prepared epoch expected by the maintenance wave.
         * @return Stable host source and device destinations for bounded DMA.
         * @throws std::logic_error When the table is not a mirrored GPU
         *         authority or the requested bank was not prepared exactly.
         *
         * This method never allocates, transfers, or changes the active bank.
         */
        [[nodiscard]] DeviceMoERuntimeBankPublicationRecipe
        preparedInactiveBankPublicationRecipe(
            int layer_idx,
            uint32_t epoch);

        /**
         * @brief Advance only the host publication recipe after GPU success.
         * @param layer_idx Model layer whose device metadata was published.
         * @param epoch Exact monotonically newer device epoch.
         * @param bank Exact bank reported by the device epoch protocol.
         * @throws std::logic_error When the acknowledgement does not match the
         *         sole prepared inactive recipe.
         *
         * No device bytes are copied.  In particular, this method must never
         * be treated as a coherence download or upload: device state remains
         * authoritative and the host retains only the next publication recipe.
         */
        void acknowledgeDevicePublishedBank(
            int layer_idx,
            uint32_t epoch,
            uint32_t bank);
        /** @copydoc IMoERuntimeTable::overlayRoutePlacementBinding */
        MoEOverlayRoutePlacementDeviceBinding
        overlayRoutePlacementBinding(int layer_idx) const override;
        /** @copydoc IMoERuntimeTable::deviceOverlayServiceTelemetry */
        DeviceMoEOverlayServiceTelemetryCell *
        deviceOverlayServiceTelemetry() const noexcept override;
        /** @copydoc IMoERuntimeTable::deviceOverlayServiceTelemetrySample */
        DeviceMoEOverlayServiceTelemetrySample *
        deviceOverlayServiceTelemetrySample(int layer_idx) const noexcept
            override;
        bool hasDeferredVerifierRouteLedgerCapacity(int layer_idx,
                                                    int token_count) const;

    private:
        DeviceId device_id_;
        int num_layers_ = 0;
        int num_experts_ = 0;
        int top_k_ = 0;
        bool mirror_to_device_ = false;
        bool collect_overlay_service_telemetry_ = false;
        int prefill_token_capacity_ = 0;
        int deferred_verifier_token_capacity_ = 0;
        std::vector<DeviceMoELayerRuntime> host_layers_;
        std::vector<DeviceMoELayerRuntime> initial_host_layers_;
        std::vector<DeviceMoELayerRuntime> empty_host_layers_;
        std::vector<uint8_t> initial_layer_captured_;
        /**
         * @brief Host-side lifecycle metadata for ordered decode-bank publication.
         *
         * This vector describes whether a publication has been enqueued; it
         * never claims that host bytes mirror live GPU bytes.  It lets graph
         * construction distinguish a reusable publication recipe from a stale
         * recipe left behind after a device-to-device reset.
         */
        std::vector<uint8_t> decode_runtime_publication_required_;
        DeviceMoELayerRuntime *device_layers_ = nullptr;
        DeviceMoELayerRuntime *device_initial_layers_ = nullptr;
        DeviceMoELayerRuntime *device_empty_layers_ = nullptr;
        /** Canonical device-local `[layer][phase]` service accumulators. */
        DeviceMoEOverlayServiceTelemetryCell
            *device_overlay_service_telemetry_ = nullptr;
        /** One zero-initialized cursor per layer across serial retained graphs. */
        DeviceMoEOverlayServiceTelemetrySample
            *device_overlay_service_samples_ = nullptr;
        void *decode_histogram_producer_stream_ = nullptr;
        /**
         * @brief Table-owned stream for accepted grouped-verifier publication.
         *
         * This stream is allocated and admitted atomically with the persistent
         * histogram banks whenever the GroupedVerifier source is selected. It
         * deliberately outlives every accepted-state graph identity and is
         * retired only after inference has stopped and borrowed graph caches
         * have released their native executables.
         */
        void *grouped_verifier_histogram_publication_stream_ = nullptr;

        /**
         * @brief Exact producer stream and its two-phase bank-rotation fences.
         *
         * The arrival closes all work submitted before a rotation request.
         * The departure is recorded after the producer has consumed the sole
         * writer-state publication. Maintenance joins both phases before it
         * snapshots or clears the frozen bank, so graph launches submitted by
         * another host thread cannot fall through a one-sided event gap.
         */
        struct RuntimeHistogramProducerStream
        {
            /**
             * @brief Declare which object owns the producer stream lifetime.
             *
             * Borrowed streams belong to graph/device contexts and must be
             * forgotten at the explicit producer-retirement edge. The grouped
             * verifier publication stream belongs to this table and remains
             * available for destruction after borrowed contexts are gone.
             */
            enum class Ownership : uint8_t
            {
                BorrowedExecutionStream = 0,
                TableOwnedStream,
            };

            void *stream = nullptr;
            void *flip_arrival_event = nullptr;
            void *flip_departure_event = nullptr;
            Ownership ownership = Ownership::BorrowedExecutionStream;
        };

        mutable std::mutex runtime_histogram_drain_mutex_;
        RuntimeExpertHistogramSourceMask runtime_histogram_sources_{};
        std::vector<RuntimeHistogramProducerStream>
            runtime_histogram_producer_streams_;
        DeviceMoERuntimeHistogramBank *device_runtime_histogram_banks_ =
            nullptr;
        uint32_t *device_runtime_histogram_active_bank_ = nullptr;
        DeviceMoERuntimeHistogramBank *host_runtime_histogram_snapshot_ =
            nullptr;
        /** Pinned identity table for admitted/quarantined states of both banks. */
        uint32_t *host_runtime_histogram_writer_states_ = nullptr;
        void *runtime_histogram_maintenance_stream_ = nullptr;
        /**
         * @brief Model-setup fence consumed by every exact producer stream.
         *
         * Histogram banks and the active-bank selector are initialized on the
         * maintenance stream.  Producers wait on this event once, during
         * topology registration, so inference never needs a host-side setup
         * synchronization.
         */
        void *runtime_histogram_initialization_event_ = nullptr;
        /**
         * @brief Reusable fence for the sole device writer-state publication.
         *
         * The maintenance stream is the only stream allowed to mutate the
         * active-bank/admission scalar. Every producer waits on this event
         * before it may read the newly published state, preventing concurrent
         * DMA writes or a graph reader racing an unordered state transition.
         */
        void *runtime_histogram_writer_state_published_event_ = nullptr;
        void *runtime_histogram_drain_complete_event_ = nullptr;
        uint32_t runtime_histogram_active_bank_host_ = 0;
        uint32_t runtime_histogram_frozen_bank_host_ = 0;
        bool runtime_histogram_drain_enabled_ = false;
        bool runtime_histogram_drain_in_flight_ = false;
        /** Typed producer-set transition enforced by the first bank rotation. */
        enum class RuntimeHistogramProducerTopology : uint8_t
        {
            Open = 0, ///< Model setup may still admit exact producer streams.
            Sealed,   ///< Every future writer must already be in the set.
        };
        RuntimeHistogramProducerTopology runtime_histogram_producer_topology_ =
            RuntimeHistogramProducerTopology::Open;
        /**
         * @brief Terminal ownership lifecycle for producer stream identities.
         *
         * This state prevents resource teardown from touching a graph-borrowed
         * stream after its device context has been destroyed. Producer
         * retirement closes and fences the DAG while all streams are live;
         * resource release later destroys only table-owned handles.
         */
        enum class RuntimeHistogramProducerLifecycle : uint8_t
        {
            CollectingProducers = 0,
            ProducersRetired,
            ResourcesReleased,
        };
        RuntimeHistogramProducerLifecycle
            runtime_histogram_producer_lifecycle_ =
                RuntimeHistogramProducerLifecycle::CollectingProducers;

        std::shared_ptr<DeviceMoESerialRouteScratchArena>
            serial_route_scratch_arena_;
        /** Keeps the stable ticket allocation alive beyond every graph/table. */
        std::shared_ptr<DeviceMoEOverlayEpochArena> overlay_epoch_arena_;
        /** Exact immutable request slot selected during model construction. */
        uint32_t overlay_epoch_ticket_slot_ = 0u;
        /** Non-owning model-lifetime source retained by the owning graph builder. */
        DeviceMoERuntimeTable *overlay_placement_source_ = nullptr;
        std::vector<DeviceMoEPrefillRouteScratchBindings>
            prefill_route_scratch_;
        int32_t *deferred_verifier_route_expert_ids_ = nullptr;
        int32_t *deferred_verifier_route_participant_ids_ = nullptr;
        uint32_t deferred_verifier_route_capacity_ = 0;

        /**
         * @brief Setup-only authority handoff for the retained runtime family.
         *
         * Collecting permits graph builders to capture each immutable layer
         * recipe. Published means one exact controller stream has made every
         * recipe device-visible; later placement changes are device-owned.
         */
        enum class InitialRuntimeFamilyLifecycle : uint8_t
        {
            Collecting = 0,
            Published,
        };

        InitialRuntimeFamilyLifecycle initial_runtime_family_lifecycle_ =
            InitialRuntimeFamilyLifecycle::Collecting;

        void validateLayerIndex(int layer_idx) const;
        void validateUpdate(int layer_idx, const MoEPlacementUpdate &update) const;
        void resetLayer(DeviceMoELayerRuntime &state) const;
        void captureInitialLayerStateIfNeeded(int layer_idx, void *stream);
        bool prefillRouteScratchAllocationHasCapacity(
            const DeviceMoEPrefillRouteScratchBindings &allocation,
            int token_capacity) const;
        void bindPrefillRouteScratchToLayer(
            int layer_idx,
            const DeviceMoEPrefillRouteScratchBindings &allocation);
        void allocateDeviceMirror();
        void releaseDeviceMirror() noexcept;
        /** Allocate and zero canonical Dynamic service accumulators. */
        void allocateOverlayServiceTelemetry();
        /** Release only a canonical table's service accumulators. */
        void releaseOverlayServiceTelemetry() noexcept;
        /** Allocate and bind model-lifetime asynchronous histogram resources. */
        void allocateRuntimeHistogramDrainResources();
        /**
         * @brief Admit one producer and order it after bank initialization.
         *
         * @param stream Exact non-null stream that writes runtime histograms.
         *
         * The caller must hold @ref runtime_histogram_drain_mutex_ and must run
         * outside a CUDA/HIP capture interval. Event allocation and the wait on
         * uncaptured model-setup work are intentionally confined to this method.
         */
        void registerRuntimeHistogramProducerStreamLocked(
            void *stream,
            RuntimeHistogramProducerStream::Ownership ownership);
        /**
         * @brief Test whether an exact stream has completed producer admission.
         *
         * @param stream Stream identity to find.
         * @return true when the stream owns a reusable flip-arrival event.
         *
         * The caller must hold @ref runtime_histogram_drain_mutex_.
         */
        [[nodiscard]] bool isRuntimeHistogramProducerStreamRegisteredLocked(
            void *stream) const noexcept;
        /**
         * @brief Close the producer DAG while every stream owner is alive.
         *
         * The caller holds @ref runtime_histogram_drain_mutex_. The method
         * records each producer's reusable arrival event, joins those events on
         * the maintenance stream, and takes the sole terminal stream fence.
         * Borrowed stream pointers are then erased while their table-owned
         * event handles remain available for later resource destruction.
         *
         * @throws std::logic_error for incomplete producer resources.
         * @throws std::runtime_error when an event edge or terminal fence fails.
         */
        void retireRuntimeHistogramProducerStreamsLocked();
        /**
         * @brief Publish one writer state through the sole maintenance stream.
         *
         * @param writer_state Valid encoded bank/admission state.
         * @param failure Receives the exact failed event edge or copy.
         * @return true when the complete asynchronous producer/publication DAG
         *         was enqueued successfully.
         *
         * The caller holds @ref runtime_histogram_drain_mutex_. Each producer
         * first records an arrival after all prior graph work. The maintenance
         * stream joins those arrivals, performs the only H2D scalar write, and
         * records @ref runtime_histogram_writer_state_published_event_. Every
         * producer waits on that publication and records its departure. The
         * maintenance stream then joins every departure before returning, so
         * its caller may snapshot/reset the frozen bank without racing work
         * interleaved by another host submission thread. No host or device
         * synchronization is introduced.
         */
        [[nodiscard]] bool publishRuntimeHistogramWriterStateLocked(
            uint32_t writer_state,
            std::string &failure);
        /**
         * @brief Destroy already-retired histogram resources at teardown.
         *
         * Borrowed producer streams must have crossed
         * @ref retireRuntimeHistogramProducerStreams before this method runs.
         * Constructor-failure cleanup may retire a set containing only
         * table-owned streams internally. Reaching normal destruction with an
         * unretired borrowed stream is a fatal lifetime violation.
         */
        void releaseRuntimeHistogramDrainResources() noexcept;
        /** Merge one completed pinned generation into the host RCU histogram. */
        bool mergeRuntimeHistogramSnapshot(
            DecodeExpertHistogram &histogram);
        void allocatePrefillRouteScratchForLayer(int layer_idx, int token_capacity);
        void releasePrefillRouteScratch() noexcept;
        void allocateDeferredVerifierRouteLedger();
        void bindDeferredVerifierRouteLedgerToLayers();
        void releaseDeferredVerifierRouteLedger() noexcept;
        void uploadLayerState(int layer_idx, void *stream);
        void uploadResetTemplatesForLayer(int layer_idx, void *stream);
        void uploadAllLayerStates();
    };

    using MoERuntimeTable = DeviceMoERuntimeTable;

} // namespace llaminar2
