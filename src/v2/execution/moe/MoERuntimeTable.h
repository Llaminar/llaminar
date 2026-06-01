/**
 * @file MoERuntimeTable.h
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include "../../tensors/TensorKernels.h"

#include <cstdint>
#include <type_traits>
#include <vector>

namespace llaminar2
{

    class DecodeExpertHistogram;

    inline constexpr uint32_t kDeviceMoEMaxExperts = 256;
    inline constexpr uint32_t kDeviceMoEMaxTopK = 16;

    enum class DeviceMoEExpertFlags : uint32_t
    {
        None = 0,
        Valid = 1u << 0,
        Resident = 1u << 1,
        Replicated = 1u << 2,
        PreferredOwner = 1u << 3,
        LocalCompute = 1u << 4,
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
        uint32_t epoch = 0;
        uint32_t expert_count = 0;
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

        int32_t *route_expert_ids = nullptr;
        float *route_weights = nullptr;
        int32_t *expert_counts = nullptr;
        int32_t *expert_offsets = nullptr;
        int32_t *grouped_token_ids = nullptr;
        float *grouped_route_weights = nullptr;
        float *grouped_gate_scratch = nullptr;
        float *grouped_up_scratch = nullptr;
        float *grouped_output_partials = nullptr;
        void *decode_scratch = nullptr;
        void *reserved_ptrs[3] = {};
        uint64_t reserved_u64[4] = {};
        uint32_t prefill_token_capacity = 0;
        uint32_t prefill_route_capacity = 0;
        uint32_t reserved_u32[2] = {};
    };

    static_assert(std::is_trivially_copyable_v<DeviceMoEExpertDescriptor>);
    static_assert(std::is_trivially_copyable_v<DeviceMoEPlacementBank>);
    static_assert(std::is_trivially_copyable_v<DeviceMoELayerRuntime>);

    struct MoEPlacementUpdate
    {
        uint32_t epoch = 0;
        uint32_t expert_count = 0;
        std::vector<DeviceMoEExpertDescriptor> experts;
        std::vector<uint8_t> local_compute_mask;
        std::vector<uint8_t> replica_role;
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
        virtual bool syncDecodeHistogramToHost(DecodeExpertHistogram &histogram,
                                               void *stream = nullptr,
                                               bool reset_runtime_counts = true) = 0;
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
        bool syncDecodeHistogramToHost(DecodeExpertHistogram &histogram,
                                       void *stream = nullptr,
                                       bool reset_runtime_counts = true) override;
        void resetDecodeHistogramCounts(void *stream = nullptr) override;
        void resetDecodeRuntimeState(void *stream = nullptr) override;
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
        DeviceMoELayerRuntime *device_layers_ = nullptr;

        struct PrefillRouteScratchAllocation
        {
            int32_t *route_expert_ids = nullptr;
            float *route_weights = nullptr;
            int32_t *expert_counts = nullptr;
            int32_t *expert_offsets = nullptr;
            int32_t *grouped_token_ids = nullptr;
            float *grouped_route_weights = nullptr;
            uint32_t token_capacity = 0;
            uint32_t route_capacity = 0;
            uint32_t expert_capacity = 0;
        };
        std::vector<PrefillRouteScratchAllocation> prefill_route_scratch_;

        void validateLayerIndex(int layer_idx) const;
        void validateUpdate(int layer_idx, const MoEPlacementUpdate &update) const;
        void resetLayer(DeviceMoELayerRuntime &state) const;
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
