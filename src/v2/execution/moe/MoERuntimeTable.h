/**
 * @file MoERuntimeTable.h
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#pragma once

#include "DeviceMoERuntimeABI.h"
#include "LeastLoadedExpertAssignment.h"

#include "../../backends/DeviceId.h"
#include "../../tensors/TensorKernels.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

namespace llaminar2
{

    class DecodeExpertHistogram;

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
        DeviceNativeVNNIMatrixDesc gate;
        DeviceNativeVNNIMatrixDesc up;
        DeviceNativeVNNIMatrixDesc down;
        int32_t logical_expert_id = -1;
        int32_t owner_participant = -1;
        int32_t local_slot = -1;
        uint32_t flags = 0;
    };

    struct DeviceMoEPlacementBank
    {
        DeviceMoEExpertDescriptor experts[kDeviceMoEMaxExperts] = {};
        uint8_t local_compute_mask[kDeviceMoEMaxExperts] = {};
        uint8_t replica_role[kDeviceMoEMaxExperts] = {};
        uint32_t resident_participant_mask[kDeviceMoEMaxExperts] = {};
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
        uint64_t decode_histogram[kDeviceMoEMaxExperts] = {};
        uint64_t decode_local_histogram[kDeviceMoEMaxExperts] = {};
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
    };

    static_assert(std::is_trivially_copyable_v<DeviceMoEExpertDescriptor>);
    static_assert(std::is_trivially_copyable_v<DeviceMoEPlacementBank>);
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
    static_assert(
        offsetof(DeviceMoELayerRuntime, deferred_verifier_route_capacity) ==
        moe_runtime_abi::kDeferredVerifierRouteCapacityOffset);
    static_assert(offsetof(DeviceMoELayerRuntime, participant_count) ==
                  moe_runtime_abi::kParticipantCountOffset);
    static_assert(
        offsetof(DeviceMoELayerRuntime,
                 current_batch_llep_movement_observed) ==
        moe_runtime_abi::kCurrentBatchLLEPMovementObservedOffset);

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
         * @brief Whether this bank includes request-lifetime transfer placement.
         *
         * Callers constructing the same logical bank for several participants
         * must publish the same value on every participant. Runtime transfer
         * apply derives it from the globally gathered transfer plan.
         */
        bool transient_placement_observed = false;
    };

    struct DeviceMoEPortableExpertRuntimeState
    {
        int32_t logical_expert_id = -1;
        int32_t owner_participant = -1;
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
         * @brief Whether exact restore requires rebuilding request-owned payloads.
         *
         * The portable record below contains logical placement only. When this
         * flag is set, one or more resident-participant bits name rolling
         * transfer-slot replicas whose pointer-bearing descriptors deliberately
         * are not serialized. Restore first publishes immutable model placement
         * plus a device transfer plan; a dedicated captured graph then recreates
         * those payloads before any restored-prefix route is assigned.
         */
        uint32_t requires_device_payload_rehydration = 0;
        std::vector<DeviceMoEPortableExpertRuntimeState> experts;
        std::vector<uint64_t> selected_histogram;
        std::vector<uint64_t> local_histogram;
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
        virtual void recordDecodeHistogramProducerStream(void *stream) = 0;
        virtual void *decodeHistogramProducerStream() const = 0;
        virtual bool syncDecodeHistogramToHost(DecodeExpertHistogram &histogram,
                                               void *stream = nullptr,
                                               bool reset_runtime_counts = true) = 0;
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
        void recordDecodeHistogramProducerStream(void *stream) override;
        void *decodeHistogramProducerStream() const override;
        bool syncDecodeHistogramToHost(DecodeExpertHistogram &histogram,
                                       void *stream = nullptr,
                                       bool reset_runtime_counts = true) override;
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
         * Portable prefix-cache state stores logical placement, local-compute
         * intent, and stable local slot ids, but deliberately does not embed
         * pointer-bearing DeviceMoELayerRuntime banks. A runtime that owns a
         * model-lifetime payload directory may provide this resolver to turn a
         * saved `(layer, expert, local_slot)` claim back into a live
         * DeviceMoEExpertDescriptor.
         *
         * Rolling transfer-slot payloads are never resolver candidates.
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
        bool restorePortableRuntimeState(const std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
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
        bool hasDeferredVerifierRouteLedgerCapacity(int layer_idx,
                                                    int token_count) const;

    private:
        DeviceId device_id_;
        int num_layers_ = 0;
        int num_experts_ = 0;
        int top_k_ = 0;
        bool mirror_to_device_ = false;
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
        void *decode_histogram_producer_stream_ = nullptr;

        std::shared_ptr<DeviceMoESerialRouteScratchArena>
            serial_route_scratch_arena_;
        std::vector<DeviceMoEPrefillRouteScratchBindings>
            prefill_route_scratch_;
        int32_t *deferred_verifier_route_expert_ids_ = nullptr;
        int32_t *deferred_verifier_route_participant_ids_ = nullptr;
        uint32_t deferred_verifier_route_capacity_ = 0;

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
