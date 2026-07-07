/**
 * @file MoERuntimeTable.h
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#pragma once

#include "LeastLoadedExpertAssignment.h"

#include "../../backends/DeviceId.h"
#include "../../tensors/TensorKernels.h"

#include <cstdint>
#include <algorithm>
#include <functional>
#include <type_traits>
#include <vector>

namespace llaminar2
{

    class DecodeExpertHistogram;

    inline constexpr uint32_t kDeviceMoEMaxExperts = 256;
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
        // reserved[0]: count of experts resident on more than one participant.
        // Decode uses this as a cheap hot-cache router-stats gate.
        uint32_t reserved[2] = {};
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
        uint32_t participant_id = 0;
        uint32_t participant_count = 1;
    };

    static_assert(std::is_trivially_copyable_v<DeviceMoEExpertDescriptor>);
    static_assert(std::is_trivially_copyable_v<DeviceMoEPlacementBank>);
    static_assert(std::is_trivially_copyable_v<DeviceMoELayerRuntime>);

    inline bool deviceMoELayerUsesTransientLocalPayload(
        const DeviceMoELayerRuntime &state) noexcept
    {
        if (state.active_bank > 1u ||
            state.active_epoch == 0u ||
            state.expert_count > kDeviceMoEMaxExperts)
        {
            return false;
        }

        const auto &bank = state.banks[state.active_bank];
        const uint32_t expert_count =
            std::min<uint32_t>(state.expert_count, kDeviceMoEMaxExperts);
        for (uint32_t expert = 0; expert < expert_count; ++expert)
        {
            if (bank.local_compute_mask[expert] != 0u &&
                hasMoEExpertFlag(bank.experts[expert].flags,
                                 DeviceMoEExpertFlags::TransferSlot))
            {
                return true;
            }
        }
        return false;
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
        std::vector<DeviceMoEPortableExpertRuntimeState> experts;
        std::vector<uint64_t> selected_histogram;
        std::vector<uint64_t> local_histogram;
    };

    class IMoERuntimeTable
    {
    public:
        virtual ~IMoERuntimeTable() = default;

        virtual DeviceMoELayerRuntime *deviceLayerState(int layer_idx) = 0;
        virtual int layerCount() const = 0;
        virtual const DeviceMoELayerRuntime &hostLayerState(int layer_idx) const = 0;
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
        void resetDecodeRuntimeState(void *stream = nullptr) override;
        bool hasInitialRuntimeState() const noexcept;
        void restoreInitialRuntimeState(void *stream = nullptr);
        void syncRuntimeStateToHost(void *stream = nullptr);
        void restoreRuntimeStateSnapshot(const DeviceMoELayerRuntime *layers,
                                         size_t layer_count,
                                         void *stream = nullptr);
        /**
         * @brief Resolver used when a portable restore must rebind local expert payloads.
         *
         * Portable prefix-cache state stores logical placement, local-compute
         * intent, and stable local slot ids, but it deliberately does not embed
         * pointer-bearing DeviceMoELayerRuntime banks.  A graph builder that owns
         * persistent transfer-slot directories may provide this resolver to turn
         * the saved `(layer, expert, local_slot)` claim back into a live
         * DeviceMoEExpertDescriptor.
         *
         * Remote-owned local replicas are transfer-slot payloads owned by the
         * graph-side slot directory, so a provided resolver is authoritative for
         * those records.  Owned local experts may still bind from an existing
         * ready runtime-bank descriptor because their payload lifetime is the
         * model/runtime table itself rather than a transient rebalance slot.
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

    private:
        DeviceId device_id_;
        int num_layers_ = 0;
        int num_experts_ = 0;
        int top_k_ = 0;
        bool mirror_to_device_ = false;
        int prefill_token_capacity_ = 0;
        std::vector<DeviceMoELayerRuntime> host_layers_;
        std::vector<DeviceMoELayerRuntime> initial_host_layers_;
        std::vector<uint8_t> initial_layer_captured_;
        DeviceMoELayerRuntime *device_layers_ = nullptr;
        void *decode_histogram_producer_stream_ = nullptr;

        struct PrefillRouteScratchAllocation
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
        std::vector<PrefillRouteScratchAllocation> prefill_route_scratch_;

        void validateLayerIndex(int layer_idx) const;
        void validateUpdate(int layer_idx, const MoEPlacementUpdate &update) const;
        void resetLayer(DeviceMoELayerRuntime &state) const;
        void captureInitialLayerStateIfNeeded(int layer_idx);
        bool prefillRouteScratchAllocationHasCapacity(const PrefillRouteScratchAllocation &allocation,
                                                      int token_capacity) const;
        void allocateDeviceMirror();
        void releaseDeviceMirror() noexcept;
        void allocatePrefillRouteScratchForLayer(int layer_idx, int token_capacity);
        void releasePrefillRouteScratch() noexcept;
        void uploadLayerState(int layer_idx, void *stream);
        void uploadAllLayerStates();
    };

    using MoERuntimeTable = DeviceMoERuntimeTable;

} // namespace llaminar2
