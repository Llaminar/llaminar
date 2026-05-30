/**
 * @file MoERuntimeTable.cpp
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#include "MoERuntimeTable.h"

#include "DecodeExpertHistogram.h"
#include "../../backends/BackendManager.h"
#include "../../utils/Logger.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace llaminar2
{
    namespace
    {
        bool descriptorReady(const DeviceMoEExpertDescriptor &desc)
        {
            return desc.gate.valid() && desc.up.valid() && desc.down.valid();
        }

        bool descriptorRequiresReadyPayload(const DeviceMoEExpertDescriptor &desc, uint8_t local_compute)
        {
            return local_compute != 0 ||
                   hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Valid) ||
                   hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::Resident) ||
                   hasMoEExpertFlag(desc.flags, DeviceMoEExpertFlags::LocalCompute);
        }

        std::string layerPrefix(int layer_idx)
        {
            return "[MoERuntimeTable] layer " + std::to_string(layer_idx) + ": ";
        }

        IBackend *mirrorBackend(DeviceId device, const std::string &what)
        {
            if (!device.is_gpu())
                throw std::runtime_error(what + ": mirrored MoE runtime tables require a GPU device, got " + device.to_string());
            IBackend *backend = getBackendFor(device);
            if (!backend)
                throw std::runtime_error(what + ": no backend available for " + device.to_string());
            return backend;
        }

        void *allocateMirror(DeviceId device, size_t bytes, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            void *ptr = backend->allocate(bytes, device.toKernelDeviceIndex());
            if (!ptr)
                throw std::runtime_error(what + ": backend allocation failed for " + std::to_string(bytes) + " bytes on " + device.to_string());
            return ptr;
        }

        void freeMirror(DeviceId device, void *ptr, const std::string &what) noexcept
        {
            if (!ptr)
                return;
            try
            {
                IBackend *backend = getBackendFor(device);
                if (!backend)
                {
                    LOG_ERROR(what << ": no backend available for " << device.to_string());
                    return;
                }
                backend->free(ptr, device.toKernelDeviceIndex());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR(what << ": backend free failed for " << device.to_string() << ": " << e.what());
            }
            catch (...)
            {
                LOG_ERROR(what << ": backend free failed for " << device.to_string() << ": unknown exception");
            }
        }

        void copyHostToMirror(DeviceId device, void *dst, const void *src,
                              size_t bytes, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            const int ordinal = device.toKernelDeviceIndex();
            const bool ok = stream
                                ? backend->hostToDeviceOnStream(dst, src, bytes, ordinal, stream)
                                : backend->hostToDevice(dst, src, bytes, ordinal, nullptr);
            if (!ok)
                throw std::runtime_error(what + ": backend H2D copy failed on " + device.to_string());
        }

        void copyMirrorToHost(DeviceId device, void *dst, const void *src,
                              size_t bytes, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            const bool ok = backend->deviceToHostFast(dst, src, bytes, device.toKernelDeviceIndex(), stream);
            if (!ok)
                throw std::runtime_error(what + ": backend D2H copy failed on " + device.to_string());
        }

        void memsetMirror(DeviceId device, void *dst, int value,
                          size_t bytes, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            if (!backend->memset(dst, value, bytes, device.toKernelDeviceIndex(), stream))
                throw std::runtime_error(what + ": backend memset failed on " + device.to_string());
        }

        void synchronizeMirror(DeviceId device, void *stream, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            const int ordinal = device.toKernelDeviceIndex();
            const bool ok = stream
                                ? backend->synchronizeStream(stream, ordinal)
                                : backend->streamSynchronize(ordinal);
            if (!ok)
                throw std::runtime_error(what + ": backend stream synchronization failed on " + device.to_string());
        }

        void *createMirrorStream(DeviceId device, const std::string &what)
        {
            IBackend *backend = mirrorBackend(device, what);
            void *stream = backend->createStream(device.toKernelDeviceIndex());
            if (!stream)
                throw std::runtime_error(what + ": backend stream creation failed on " + device.to_string());
            return stream;
        }

        void destroyMirrorStream(DeviceId device, void *stream, const std::string &what) noexcept
        {
            if (!stream)
                return;
            try
            {
                IBackend *backend = getBackendFor(device);
                if (!backend)
                {
                    LOG_ERROR(what << ": no backend available for " << device.to_string());
                    return;
                }
                backend->destroyStream(stream, device.toKernelDeviceIndex());
            }
            catch (const std::exception &e)
            {
                LOG_ERROR(what << ": backend stream destroy failed for " << device.to_string() << ": " << e.what());
            }
            catch (...)
            {
                LOG_ERROR(what << ": backend stream destroy failed for " << device.to_string() << ": unknown exception");
            }
        }

        uint32_t checkedRouteCapacity(int token_capacity, int top_k)
        {
            if (token_capacity < 0)
                throw std::invalid_argument("[MoERuntimeTable] prefill token capacity must be non-negative");
            const uint64_t route_capacity = static_cast<uint64_t>(token_capacity) * static_cast<uint64_t>(top_k);
            if (route_capacity > std::numeric_limits<uint32_t>::max())
                throw std::invalid_argument("[MoERuntimeTable] prefill route capacity exceeds uint32_t range");
            return static_cast<uint32_t>(route_capacity);
        }

    } // namespace

    DeviceMoERuntimeTable::DeviceMoERuntimeTable(Config config)
        : device_id_(config.device_id),
          num_layers_(config.num_layers),
          num_experts_(config.num_experts),
          top_k_(config.top_k),
          mirror_to_device_(config.mirror_to_device),
          prefill_token_capacity_(config.prefill_token_capacity)
    {
        if (!device_id_.is_valid())
            throw std::invalid_argument("[MoERuntimeTable] device_id must be valid");
        if (num_layers_ <= 0)
            throw std::invalid_argument("[MoERuntimeTable] num_layers must be positive");
        if (num_experts_ <= 0 || num_experts_ > static_cast<int>(kDeviceMoEMaxExperts))
            throw std::invalid_argument("[MoERuntimeTable] num_experts must be in [1, " +
                                        std::to_string(kDeviceMoEMaxExperts) + "]");
        if (top_k_ <= 0 || top_k_ > static_cast<int>(kDeviceMoEMaxTopK))
            throw std::invalid_argument("[MoERuntimeTable] top_k must be in [1, " +
                                        std::to_string(kDeviceMoEMaxTopK) + "]");
        if (mirror_to_device_ && !device_id_.is_gpu())
            throw std::runtime_error("[MoERuntimeTable] device mirroring requires a GPU device");
        if (prefill_token_capacity_ < 0)
            throw std::invalid_argument("[MoERuntimeTable] prefill_token_capacity must be non-negative");
        if (prefill_token_capacity_ > 0 && !mirror_to_device_)
            throw std::runtime_error("[MoERuntimeTable] prefill route scratch requires a mirrored GPU runtime table");
        (void)checkedRouteCapacity(prefill_token_capacity_, top_k_);

        host_layers_.resize(static_cast<size_t>(num_layers_));
        for (auto &state : host_layers_)
            resetLayer(state);

        if (mirror_to_device_)
        {
            allocateDeviceMirror();
            if (prefill_token_capacity_ > 0)
            {
                prefill_route_scratch_.resize(host_layers_.size());
                for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
                    allocatePrefillRouteScratchForLayer(layer_idx, prefill_token_capacity_);
            }
            uploadAllLayerStates();
        }
    }

    DeviceMoERuntimeTable::DeviceMoERuntimeTable(DeviceId device_id,
                                                 int num_layers,
                                                 int num_experts,
                                                 int top_k,
                                                 bool mirror_to_device)
        : DeviceMoERuntimeTable(Config{.device_id = device_id,
                                       .num_layers = num_layers,
                                       .num_experts = num_experts,
                                       .top_k = top_k,
                                       .mirror_to_device = mirror_to_device})
    {
    }

    DeviceMoERuntimeTable::~DeviceMoERuntimeTable()
    {
        releasePrefillRouteScratch();
        releaseDeviceMirror();
    }

    DeviceMoELayerRuntime *DeviceMoERuntimeTable::deviceLayerState(int layer_idx)
    {
        validateLayerIndex(layer_idx);
        if (mirror_to_device_)
            return device_layers_ + layer_idx;
        return host_layers_.data() + layer_idx;
    }

    DeviceMoELayerRuntime &DeviceMoERuntimeTable::hostLayerState(int layer_idx)
    {
        validateLayerIndex(layer_idx);
        return host_layers_[static_cast<size_t>(layer_idx)];
    }

    const DeviceMoELayerRuntime &DeviceMoERuntimeTable::hostLayerState(int layer_idx) const
    {
        validateLayerIndex(layer_idx);
        return host_layers_[static_cast<size_t>(layer_idx)];
    }

    bool DeviceMoERuntimeTable::hasPrefillRouteScratchCapacity(int layer_idx, int token_count) const
    {
        validateLayerIndex(layer_idx);
        if (token_count <= 0)
            return false;
        const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        const uint32_t route_count = checkedRouteCapacity(token_count, top_k_);
        return state.prefill_token_capacity >= static_cast<uint32_t>(token_count) &&
               state.prefill_route_capacity >= route_count &&
               state.route_expert_ids &&
               state.route_weights &&
               state.expert_counts &&
               state.expert_offsets &&
               state.grouped_token_ids &&
               state.grouped_route_weights;
    }

    bool DeviceMoERuntimeTable::syncDecodeHistogramToHost(
        DecodeExpertHistogram &histogram,
        void *stream,
        bool reset_runtime_counts)
    {
        const auto &hist_config = histogram.config();
        if (hist_config.num_layers != num_layers_ ||
            hist_config.num_experts != num_experts_ ||
            hist_config.top_k != top_k_)
        {
            LOG_ERROR("[MoERuntimeTable] decode histogram config mismatch: table layers="
                      << num_layers_ << " experts=" << num_experts_ << " top_k=" << top_k_
                      << " histogram layers=" << hist_config.num_layers
                      << " experts=" << hist_config.num_experts
                      << " top_k=" << hist_config.top_k);
            return false;
        }

        std::vector<uint64_t> counts(static_cast<size_t>(num_layers_) * static_cast<size_t>(num_experts_), 0);

        if (!mirror_to_device_)
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                auto *dst = counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                std::copy(state.decode_histogram,
                          state.decode_histogram + num_experts_,
                          dst);
            }
        }
        else
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto *src = device_layers_[layer_idx].decode_histogram;
                auto *dst = counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                copyMirrorToHost(device_id_, dst, src,
                                 static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                                 stream,
                                 layerPrefix(layer_idx) + "decode histogram D2H");
            }
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto *layer_counts = counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            histogram.mergeLayerCounts(layer_idx, layer_counts, num_experts_, /*count_window_tokens=*/false);
        }

        if (!reset_runtime_counts)
            return true;

        for (auto &state : host_layers_)
            std::fill(state.decode_histogram, state.decode_histogram + num_experts_, 0ULL);

        if (mirror_to_device_)
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                auto *dst = device_layers_[layer_idx].decode_histogram;
                memsetMirror(device_id_, dst, 0,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode histogram reset");
            }
            synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode histogram reset sync");
        }

        return true;
    }

    void DeviceMoERuntimeTable::resetDecodeHistogramCounts(void *stream)
    {
        for (auto &state : host_layers_)
            std::fill(state.decode_histogram, state.decode_histogram + num_experts_, 0ULL);

        if (!mirror_to_device_)
            return;

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto *dst = device_layers_[layer_idx].decode_histogram;
            memsetMirror(device_id_, dst, 0,
                         static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                         stream,
                         layerPrefix(layer_idx) + "decode histogram reset");
        }
        synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode histogram reset sync");
    }

    void DeviceMoERuntimeTable::resetDecodeRuntimeState(void *stream)
    {
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            const auto &scratch = prefill_route_scratch_.empty()
                                      ? PrefillRouteScratchAllocation{}
                                      : prefill_route_scratch_[static_cast<size_t>(layer_idx)];

            resetLayer(state);
            state.route_expert_ids = scratch.route_expert_ids;
            state.route_weights = scratch.route_weights;
            state.expert_counts = scratch.expert_counts;
            state.expert_offsets = scratch.expert_offsets;
            state.grouped_token_ids = scratch.grouped_token_ids;
            state.grouped_route_weights = scratch.grouped_route_weights;
            state.prefill_token_capacity = scratch.token_capacity;
            state.prefill_route_capacity = scratch.route_capacity;
        }

        if (!mirror_to_device_)
            return;

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            uploadLayerState(layer_idx, stream);
        synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode runtime reset sync");
    }

    void DeviceMoERuntimeTable::ensurePrefillRouteScratchCapacity(int token_capacity, void *stream)
    {
        if (token_capacity <= 0)
            throw std::invalid_argument("[MoERuntimeTable] prefill route scratch token_capacity must be positive");
        if (!mirror_to_device_ || !device_id_.is_gpu())
            throw std::runtime_error("[MoERuntimeTable] prefill route scratch requires a mirrored GPU runtime table");
        (void)checkedRouteCapacity(token_capacity, top_k_);

        if (static_cast<int>(prefill_route_scratch_.size()) != num_layers_)
            prefill_route_scratch_.resize(static_cast<size_t>(num_layers_));

        bool changed = false;
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto &allocation = prefill_route_scratch_[static_cast<size_t>(layer_idx)];
            if (!prefillRouteScratchAllocationHasCapacity(allocation, token_capacity))
            {
                allocatePrefillRouteScratchForLayer(layer_idx, token_capacity);
                changed = true;
            }
        }

        if (changed)
        {
            prefill_token_capacity_ = std::max(prefill_token_capacity_, token_capacity);
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
                uploadLayerState(layer_idx, stream);
            synchronizeMirror(device_id_, stream, "[MoERuntimeTable] prefill route scratch upload sync");
        }
    }

    bool DeviceMoERuntimeTable::prepareInactiveBank(int layer_idx, const MoEPlacementUpdate &update)
    {
        validateLayerIndex(layer_idx);
        validateUpdate(layer_idx, update);

        auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        const uint32_t inactive_bank = 1u - state.active_bank;
        auto &bank = state.banks[inactive_bank];
        bank = {};
        bank.epoch = update.epoch;
        bank.expert_count = update.expert_count;

        for (uint32_t expert = 0; expert < update.expert_count; ++expert)
        {
            bank.experts[expert] = update.experts[expert];
            bank.local_compute_mask[expert] = update.local_compute_mask[expert];
            bank.replica_role[expert] = update.replica_role[expert];
        }

        return true;
    }

    bool DeviceMoERuntimeTable::flipActiveBank(int layer_idx, uint32_t epoch, void *stream)
    {
        validateLayerIndex(layer_idx);
        auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        const uint32_t inactive_bank = 1u - state.active_bank;
        const auto &prepared_bank = state.banks[inactive_bank];

        if (prepared_bank.epoch == 0 || prepared_bank.expert_count != static_cast<uint32_t>(num_experts_))
            throw std::runtime_error(layerPrefix(layer_idx) + "inactive bank has not been prepared");
        if (prepared_bank.epoch != epoch)
            throw std::invalid_argument(layerPrefix(layer_idx) + "flip epoch does not match prepared inactive bank epoch");
        if (epoch <= state.active_epoch)
            throw std::invalid_argument(layerPrefix(layer_idx) + "flip epoch must increase monotonically");

        state.active_bank = inactive_bank;
        state.active_epoch = epoch;

        if (mirror_to_device_)
            uploadLayerState(layer_idx, stream);

        return true;
    }

    void DeviceMoERuntimeTable::validateLayerIndex(int layer_idx) const
    {
        if (layer_idx < 0 || layer_idx >= num_layers_)
            throw std::out_of_range("[MoERuntimeTable] layer index out of range: " + std::to_string(layer_idx));
    }

    void DeviceMoERuntimeTable::validateUpdate(int layer_idx, const MoEPlacementUpdate &update) const
    {
        if (update.epoch == 0)
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update epoch must be non-zero");
        const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        if (update.epoch <= state.active_epoch)
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update epoch must be newer than active epoch");
        if (update.expert_count != static_cast<uint32_t>(num_experts_))
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update expert_count must match table expert_count");
        if (update.experts.size() != update.expert_count ||
            update.local_compute_mask.size() != update.expert_count ||
            update.replica_role.size() != update.expert_count)
        {
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update vectors must match expert_count");
        }

        for (uint32_t expert = 0; expert < update.expert_count; ++expert)
        {
            const auto &desc = update.experts[expert];
            if (desc.logical_expert_id != -1 && desc.logical_expert_id != static_cast<int32_t>(expert))
                throw std::invalid_argument(layerPrefix(layer_idx) + "descriptor logical_expert_id must match table index");
            if (desc.local_slot < -1)
                throw std::invalid_argument(layerPrefix(layer_idx) + "descriptor local_slot must be -1 or non-negative");
            if (update.local_compute_mask[expert] > 1)
                throw std::invalid_argument(layerPrefix(layer_idx) + "local_compute_mask entries must be 0 or 1");
            if (update.replica_role[expert] > static_cast<uint8_t>(DeviceMoEReplicaRole::PreferredReplica))
                throw std::invalid_argument(layerPrefix(layer_idx) + "replica_role entries must be valid DeviceMoEReplicaRole values");

            if (descriptorRequiresReadyPayload(desc, update.local_compute_mask[expert]))
            {
                if (desc.logical_expert_id != static_cast<int32_t>(expert))
                    throw std::invalid_argument(layerPrefix(layer_idx) + "active descriptor must name its logical expert");
                if (!descriptorReady(desc))
                    throw std::invalid_argument(layerPrefix(layer_idx) + "active descriptor must include ready gate/up/down payload descriptors");
            }
        }
    }

    void DeviceMoERuntimeTable::resetLayer(DeviceMoELayerRuntime &state) const
    {
        state = {};
        state.expert_count = static_cast<uint32_t>(num_experts_);
        state.top_k = static_cast<uint32_t>(top_k_);
        state.banks[0].expert_count = static_cast<uint32_t>(num_experts_);
        state.banks[1].expert_count = static_cast<uint32_t>(num_experts_);
    }

    bool DeviceMoERuntimeTable::prefillRouteScratchAllocationHasCapacity(
        const PrefillRouteScratchAllocation &allocation,
        int token_capacity) const
    {
        const uint32_t route_capacity = checkedRouteCapacity(token_capacity, top_k_);
        return allocation.token_capacity >= static_cast<uint32_t>(token_capacity) &&
               allocation.route_capacity >= route_capacity &&
               allocation.expert_capacity >= static_cast<uint32_t>(num_experts_) &&
               allocation.route_expert_ids &&
               allocation.route_weights &&
               allocation.expert_counts &&
               allocation.expert_offsets &&
               allocation.grouped_token_ids &&
               allocation.grouped_route_weights;
    }

    void DeviceMoERuntimeTable::allocatePrefillRouteScratchForLayer(int layer_idx, int token_capacity)
    {
        validateLayerIndex(layer_idx);
        if (!mirror_to_device_ || !device_id_.is_gpu())
            throw std::runtime_error(layerPrefix(layer_idx) + "prefill route scratch requires a mirrored GPU runtime table");
        if (token_capacity <= 0)
            throw std::invalid_argument(layerPrefix(layer_idx) + "prefill route scratch token capacity must be positive");

        if (static_cast<int>(prefill_route_scratch_.size()) != num_layers_)
            prefill_route_scratch_.resize(static_cast<size_t>(num_layers_));

        auto &allocation = prefill_route_scratch_[static_cast<size_t>(layer_idx)];
        if (prefillRouteScratchAllocationHasCapacity(allocation, token_capacity))
            return;

        auto free_allocation = [&](PrefillRouteScratchAllocation &scratch) noexcept
        {
            if (scratch.route_expert_ids)
                freeMirror(device_id_, scratch.route_expert_ids, layerPrefix(layer_idx) + "free prefill route_expert_ids");
            if (scratch.route_weights)
                freeMirror(device_id_, scratch.route_weights, layerPrefix(layer_idx) + "free prefill route_weights");
            if (scratch.expert_counts)
                freeMirror(device_id_, scratch.expert_counts, layerPrefix(layer_idx) + "free prefill expert_counts");
            if (scratch.expert_offsets)
                freeMirror(device_id_, scratch.expert_offsets, layerPrefix(layer_idx) + "free prefill expert_offsets");
            if (scratch.grouped_token_ids)
                freeMirror(device_id_, scratch.grouped_token_ids, layerPrefix(layer_idx) + "free prefill grouped_token_ids");
            if (scratch.grouped_route_weights)
                freeMirror(device_id_, scratch.grouped_route_weights, layerPrefix(layer_idx) + "free prefill grouped_route_weights");
            scratch = {};
        };

        free_allocation(allocation);

        const uint32_t route_capacity = checkedRouteCapacity(token_capacity, top_k_);
        auto allocate = [&](auto **ptr, size_t count, const char *name)
        {
            using Pointer = std::remove_pointer_t<std::remove_pointer_t<decltype(ptr)>>;
            *ptr = static_cast<Pointer *>(allocateMirror(device_id_, count * sizeof(Pointer),
                                                         layerPrefix(layer_idx) + "allocation failed for " + name));
        };

        try
        {
            allocate(&allocation.route_expert_ids, route_capacity, "prefill route_expert_ids");
            allocate(&allocation.route_weights, route_capacity, "prefill route_weights");
            allocate(&allocation.expert_counts, static_cast<size_t>(num_experts_), "prefill expert_counts");
            allocate(&allocation.expert_offsets, static_cast<size_t>(num_experts_), "prefill expert_offsets");
            allocate(&allocation.grouped_token_ids, route_capacity, "prefill grouped_token_ids");
            allocate(&allocation.grouped_route_weights, route_capacity, "prefill grouped_route_weights");
        }
        catch (...)
        {
            free_allocation(allocation);
            throw;
        }

        allocation.token_capacity = static_cast<uint32_t>(token_capacity);
        allocation.route_capacity = route_capacity;
        allocation.expert_capacity = static_cast<uint32_t>(num_experts_);

        auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        state.route_expert_ids = allocation.route_expert_ids;
        state.route_weights = allocation.route_weights;
        state.expert_counts = allocation.expert_counts;
        state.expert_offsets = allocation.expert_offsets;
        state.grouped_token_ids = allocation.grouped_token_ids;
        state.grouped_route_weights = allocation.grouped_route_weights;
        state.prefill_token_capacity = allocation.token_capacity;
        state.prefill_route_capacity = allocation.route_capacity;
    }

    void DeviceMoERuntimeTable::releasePrefillRouteScratch() noexcept
    {
        if (prefill_route_scratch_.empty())
            return;
        for (auto &allocation : prefill_route_scratch_)
        {
            if (allocation.route_expert_ids)
                freeMirror(device_id_, allocation.route_expert_ids, "[MoERuntimeTable] free prefill route_expert_ids");
            if (allocation.route_weights)
                freeMirror(device_id_, allocation.route_weights, "[MoERuntimeTable] free prefill route_weights");
            if (allocation.expert_counts)
                freeMirror(device_id_, allocation.expert_counts, "[MoERuntimeTable] free prefill expert_counts");
            if (allocation.expert_offsets)
                freeMirror(device_id_, allocation.expert_offsets, "[MoERuntimeTable] free prefill expert_offsets");
            if (allocation.grouped_token_ids)
                freeMirror(device_id_, allocation.grouped_token_ids, "[MoERuntimeTable] free prefill grouped_token_ids");
            if (allocation.grouped_route_weights)
                freeMirror(device_id_, allocation.grouped_route_weights, "[MoERuntimeTable] free prefill grouped_route_weights");
            allocation = {};
        }
        prefill_route_scratch_.clear();
    }

    void DeviceMoERuntimeTable::allocateDeviceMirror()
    {
        const size_t bytes = host_layers_.size() * sizeof(DeviceMoELayerRuntime);
        device_layers_ = static_cast<DeviceMoELayerRuntime *>(
            allocateMirror(device_id_, bytes, "[MoERuntimeTable] runtime table mirror allocation"));
    }

    void DeviceMoERuntimeTable::releaseDeviceMirror() noexcept
    {
        if (!device_layers_)
            return;
        freeMirror(device_id_, device_layers_, "[MoERuntimeTable] free runtime table mirror");
        device_layers_ = nullptr;
    }

    void DeviceMoERuntimeTable::uploadLayerState(int layer_idx, void *stream)
    {
        auto *dst = device_layers_ + layer_idx;
        auto *src = host_layers_.data() + layer_idx;
        copyHostToMirror(device_id_, dst, src, sizeof(DeviceMoELayerRuntime), stream,
                         layerPrefix(layer_idx) + "runtime table upload");
    }

    void DeviceMoERuntimeTable::uploadAllLayerStates()
    {
        // Create a dedicated one-shot stream for the bulk initialization upload.
        // We avoid the default stream (nullptr/0) to maintain stream hygiene --
        // all async GPU work must use an explicit stream for correctness and overlap.
        void *init_stream = createMirrorStream(device_id_, "[MoERuntimeTable] runtime table initial upload stream");
        const size_t bytes = host_layers_.size() * sizeof(DeviceMoELayerRuntime);
        try
        {
            copyHostToMirror(device_id_, device_layers_, host_layers_.data(), bytes, init_stream,
                             "[MoERuntimeTable] runtime table initial upload");
            synchronizeMirror(device_id_, init_stream, "[MoERuntimeTable] runtime table initial upload sync");
            destroyMirrorStream(device_id_, init_stream, "[MoERuntimeTable] runtime table initial upload stream destroy");
        }
        catch (...)
        {
            destroyMirrorStream(device_id_, init_stream, "[MoERuntimeTable] runtime table initial upload stream destroy");
            throw;
        }
    }

} // namespace llaminar2