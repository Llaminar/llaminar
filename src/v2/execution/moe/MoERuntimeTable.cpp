/**
 * @file MoERuntimeTable.cpp
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#include "MoERuntimeTable.h"

#include "DecodeExpertHistogram.h"
#include "../../utils/Logger.h"

#ifdef HAVE_ROCM
#include "../../backends/rocm/HipDeviceGuard.h"
#include <hip/hip_runtime.h>
#endif

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

#ifdef HAVE_ROCM
        void throwOnHipError(hipError_t err, const std::string &what)
        {
            if (err == hipSuccess)
                return;
            const std::string message = what + ": " + hipGetErrorString(err);
            LOG_ERROR(message);
            throw std::runtime_error(message);
        }

        void synchronizeHipStream(void *stream, const std::string &what)
        {
            auto hip_stream = static_cast<hipStream_t>(stream);
            throwOnHipError(hipStreamSynchronize(hip_stream), what);
        }
#endif

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
        if (mirror_to_device_ && !device_id_.is_rocm())
            throw std::runtime_error("[MoERuntimeTable] device mirroring is currently implemented only for ROCm devices");
        if (prefill_token_capacity_ < 0)
            throw std::invalid_argument("[MoERuntimeTable] prefill_token_capacity must be non-negative");
        if (prefill_token_capacity_ > 0 && !mirror_to_device_)
            throw std::runtime_error("[MoERuntimeTable] prefill route scratch requires a mirrored ROCm runtime table");
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
#ifdef HAVE_ROCM
            const hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.rocm_ordinal()));
            throwOnHipError(set_err, "[MoERuntimeTable] hipSetDevice failed for decode histogram sync");
            auto hip_stream = static_cast<hipStream_t>(stream);
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto *src = device_layers_[layer_idx].decode_histogram;
                auto *dst = counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                throwOnHipError(hipMemcpyAsync(dst,
                                               src,
                                               static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                                               hipMemcpyDeviceToHost,
                                               hip_stream),
                                layerPrefix(layer_idx) + "hipMemcpyAsync failed for decode histogram D2H");
            }
            throwOnHipError(hipStreamSynchronize(hip_stream),
                            "[MoERuntimeTable] hipStreamSynchronize failed for decode histogram D2H");
#else
            (void)stream;
            (void)reset_runtime_counts;
            LOG_ERROR("[MoERuntimeTable] ROCm decode histogram sync requested but HAVE_ROCM is not enabled");
            return false;
#endif
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
#ifdef HAVE_ROCM
            auto hip_stream = static_cast<hipStream_t>(stream);
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                auto *dst = device_layers_[layer_idx].decode_histogram;
                throwOnHipError(hipMemsetAsync(dst,
                                               0,
                                               static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                                               hip_stream),
                                layerPrefix(layer_idx) + "hipMemsetAsync failed for decode histogram reset");
            }
            throwOnHipError(hipStreamSynchronize(hip_stream),
                            "[MoERuntimeTable] hipStreamSynchronize failed for decode histogram reset");
#endif
        }

        return true;
    }

    void DeviceMoERuntimeTable::resetDecodeHistogramCounts(void *stream)
    {
        for (auto &state : host_layers_)
            std::fill(state.decode_histogram, state.decode_histogram + num_experts_, 0ULL);

        if (!mirror_to_device_)
            return;

#ifdef HAVE_ROCM
        const hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.rocm_ordinal()));
        throwOnHipError(set_err, "[MoERuntimeTable] hipSetDevice failed for decode histogram reset");
        auto hip_stream = static_cast<hipStream_t>(stream);
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto *dst = device_layers_[layer_idx].decode_histogram;
            throwOnHipError(hipMemsetAsync(dst,
                                           0,
                                           static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                                           hip_stream),
                            layerPrefix(layer_idx) + "hipMemsetAsync failed for decode histogram reset");
        }
        throwOnHipError(hipStreamSynchronize(hip_stream),
                        "[MoERuntimeTable] hipStreamSynchronize failed for decode histogram reset");
#else
        (void)stream;
        throw std::runtime_error("[MoERuntimeTable] ROCm decode histogram reset requested but HAVE_ROCM is not enabled");
#endif
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

#ifdef HAVE_ROCM
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            uploadLayerState(layer_idx, stream);
        synchronizeHipStream(stream, "[MoERuntimeTable] hipStreamSynchronize failed for decode runtime reset");
#else
        (void)stream;
        throw std::runtime_error("[MoERuntimeTable] ROCm decode runtime reset requested but HAVE_ROCM is not enabled");
#endif
    }

    void DeviceMoERuntimeTable::ensurePrefillRouteScratchCapacity(int token_capacity, void *stream)
    {
        if (token_capacity <= 0)
            throw std::invalid_argument("[MoERuntimeTable] prefill route scratch token_capacity must be positive");
        if (!mirror_to_device_ || !device_id_.is_rocm())
            throw std::runtime_error("[MoERuntimeTable] prefill route scratch requires a mirrored ROCm runtime table");
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
#ifdef HAVE_ROCM
            synchronizeHipStream(stream, "[MoERuntimeTable] hipStreamSynchronize failed for prefill route scratch upload");
#endif
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
        if (!mirror_to_device_ || !device_id_.is_rocm())
            throw std::runtime_error(layerPrefix(layer_idx) + "prefill route scratch requires a mirrored ROCm runtime table");
        if (token_capacity <= 0)
            throw std::invalid_argument(layerPrefix(layer_idx) + "prefill route scratch token capacity must be positive");

#ifdef HAVE_ROCM
        const hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.rocm_ordinal()));
        throwOnHipError(set_err, layerPrefix(layer_idx) + "hipSetDevice failed for prefill route scratch allocation");

        if (static_cast<int>(prefill_route_scratch_.size()) != num_layers_)
            prefill_route_scratch_.resize(static_cast<size_t>(num_layers_));

        auto &allocation = prefill_route_scratch_[static_cast<size_t>(layer_idx)];
        if (prefillRouteScratchAllocationHasCapacity(allocation, token_capacity))
            return;

        auto free_allocation = [](PrefillRouteScratchAllocation &scratch) noexcept
        {
            if (scratch.route_expert_ids)
                (void)hipFree(scratch.route_expert_ids);
            if (scratch.route_weights)
                (void)hipFree(scratch.route_weights);
            if (scratch.expert_counts)
                (void)hipFree(scratch.expert_counts);
            if (scratch.expert_offsets)
                (void)hipFree(scratch.expert_offsets);
            if (scratch.grouped_token_ids)
                (void)hipFree(scratch.grouped_token_ids);
            if (scratch.grouped_route_weights)
                (void)hipFree(scratch.grouped_route_weights);
            scratch = {};
        };

        free_allocation(allocation);

        const uint32_t route_capacity = checkedRouteCapacity(token_capacity, top_k_);
        auto allocate = [&](auto **ptr, size_t count, const char *name)
        {
            using Pointer = std::remove_pointer_t<std::remove_pointer_t<decltype(ptr)>>;
            throwOnHipError(hipMalloc(reinterpret_cast<void **>(ptr), count * sizeof(Pointer)),
                            layerPrefix(layer_idx) + "hipMalloc failed for " + name);
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
#else
        (void)token_capacity;
        throw std::runtime_error(layerPrefix(layer_idx) + "ROCm prefill route scratch requested but HAVE_ROCM is not enabled");
#endif
    }

    void DeviceMoERuntimeTable::releasePrefillRouteScratch() noexcept
    {
        if (prefill_route_scratch_.empty())
            return;
#ifdef HAVE_ROCM
        (void)HipDeviceGuard::setDevice(device_id_.rocm_ordinal());
        for (auto &allocation : prefill_route_scratch_)
        {
            if (allocation.route_expert_ids)
                (void)hipFree(allocation.route_expert_ids);
            if (allocation.route_weights)
                (void)hipFree(allocation.route_weights);
            if (allocation.expert_counts)
                (void)hipFree(allocation.expert_counts);
            if (allocation.expert_offsets)
                (void)hipFree(allocation.expert_offsets);
            if (allocation.grouped_token_ids)
                (void)hipFree(allocation.grouped_token_ids);
            if (allocation.grouped_route_weights)
                (void)hipFree(allocation.grouped_route_weights);
            allocation = {};
        }
#endif
        prefill_route_scratch_.clear();
    }

    void DeviceMoERuntimeTable::allocateDeviceMirror()
    {
#ifdef HAVE_ROCM
        const hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.rocm_ordinal()));
        throwOnHipError(set_err, "[MoERuntimeTable] hipSetDevice failed for ROCm mirror allocation");
        const size_t bytes = host_layers_.size() * sizeof(DeviceMoELayerRuntime);
        throwOnHipError(hipMalloc(reinterpret_cast<void **>(&device_layers_), bytes),
                        "[MoERuntimeTable] hipMalloc failed for ROCm runtime table mirror");
#else
        throw std::runtime_error("[MoERuntimeTable] ROCm device mirroring requested but HAVE_ROCM is not enabled");
#endif
    }

    void DeviceMoERuntimeTable::releaseDeviceMirror() noexcept
    {
        if (!device_layers_)
            return;
#ifdef HAVE_ROCM
        (void)HipDeviceGuard::setDevice(device_id_.rocm_ordinal());
        const hipError_t err = hipFree(device_layers_);
        if (err != hipSuccess)
            LOG_ERROR("[MoERuntimeTable] hipFree failed for ROCm runtime table mirror: " << hipGetErrorString(err));
#endif
        device_layers_ = nullptr;
    }

    void DeviceMoERuntimeTable::uploadLayerState(int layer_idx, void *stream)
    {
#ifdef HAVE_ROCM
        const hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.rocm_ordinal()));
        throwOnHipError(set_err, "[MoERuntimeTable] hipSetDevice failed for ROCm runtime table upload");
        auto *dst = device_layers_ + layer_idx;
        auto *src = host_layers_.data() + layer_idx;
        auto hip_stream = static_cast<hipStream_t>(stream);
        throwOnHipError(hipMemcpyAsync(dst, src, sizeof(DeviceMoELayerRuntime), hipMemcpyHostToDevice, hip_stream),
                        "[MoERuntimeTable] hipMemcpyAsync failed for ROCm runtime table upload");
#else
        (void)layer_idx;
        (void)stream;
        throw std::runtime_error("[MoERuntimeTable] ROCm device mirroring requested but HAVE_ROCM is not enabled");
#endif
    }

    void DeviceMoERuntimeTable::uploadAllLayerStates()
    {
#ifdef HAVE_ROCM
        const hipError_t set_err = static_cast<hipError_t>(HipDeviceGuard::setDevice(device_id_.rocm_ordinal()));
        throwOnHipError(set_err, "[MoERuntimeTable] hipSetDevice failed for ROCm runtime table initial upload");

        // Create a dedicated one-shot stream for the bulk initialization upload.
        // We avoid the default stream (nullptr/0) to maintain stream hygiene —
        // all async GPU work must use an explicit stream for correctness and overlap.
        hipStream_t init_stream;
        throwOnHipError(hipStreamCreate(&init_stream),
                        "[MoERuntimeTable] hipStreamCreate failed for ROCm runtime table initial upload");

        const size_t bytes = host_layers_.size() * sizeof(DeviceMoELayerRuntime);
        throwOnHipError(hipMemcpyAsync(device_layers_, host_layers_.data(), bytes, hipMemcpyHostToDevice, init_stream),
                        "[MoERuntimeTable] hipMemcpyAsync failed for ROCm runtime table initial upload");
        throwOnHipError(hipStreamSynchronize(init_stream),
                        "[MoERuntimeTable] hipStreamSynchronize failed for ROCm runtime table initial upload");
        throwOnHipError(hipStreamDestroy(init_stream),
                        "[MoERuntimeTable] hipStreamDestroy failed for ROCm runtime table initial upload");
#else
        throw std::runtime_error("[MoERuntimeTable] ROCm device mirroring requested but HAVE_ROCM is not enabled");
#endif
    }

} // namespace llaminar2