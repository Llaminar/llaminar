/**
 * @file MoERuntimeTable.cpp
 * @brief Stable graph-facing MoE placement runtime tables.
 */

#include "MoERuntimeTable.h"

#include "DecodeExpertHistogram.h"
#include "../../backends/BackendManager.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
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

        uint32_t portableMoEExpertFlags(uint32_t flags) noexcept
        {
            return flags & ~toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
        }

        uint32_t clearPayloadBearingMoEExpertFlags(uint32_t flags) noexcept
        {
            return flags & ~(toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
                             toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
                             toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute) |
                             toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot));
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

        uint32_t participantBit(uint32_t participant)
        {
            return 1u << participant;
        }

        uint32_t participantMaskLimit(uint32_t participant_count)
        {
            return (1u << participant_count) - 1u;
        }

        uint32_t participantMaskCount(uint32_t mask) noexcept
        {
            uint32_t count = 0;
            while (mask != 0u)
            {
                mask &= (mask - 1u);
                ++count;
            }
            return count;
        }

        uint32_t synthesizedResidentParticipantMask(
            const MoEPlacementUpdate &update,
            uint32_t expert)
        {
            uint32_t mask = 0;
            const auto &desc = update.experts[expert];
            if (desc.owner_participant >= 0 &&
                desc.owner_participant < static_cast<int32_t>(update.participant_count))
            {
                mask |= participantBit(static_cast<uint32_t>(desc.owner_participant));
            }
            if (update.local_compute_mask[expert] != 0u)
                mask |= participantBit(update.participant_id);
            return mask;
        }

        uint32_t residentParticipantMaskForUpdate(
            const MoEPlacementUpdate &update,
            uint32_t expert)
        {
            if (!update.resident_participant_mask.empty())
                return update.resident_participant_mask[expert];
            return synthesizedResidentParticipantMask(update, expert);
        }

        void resetRouterHotCacheCounters(DeviceMoELayerRuntime &state) noexcept
        {
            state.router_hot_cache_eligible_dispatches = 0;
            state.router_hot_cache_used_dispatches = 0;
            state.router_hot_cache_improved_dispatches = 0;
            state.router_hot_cache_default_load_spread_total = 0;
            state.router_hot_cache_actual_load_spread_total = 0;
            state.router_hot_cache_load_spread_improvement_total = 0;
            state.router_hot_cache_active_dispatches = 0;
            state.router_hot_cache_miss_dispatches = 0;
            state.router_hot_cache_selected_expert_slots = 0;
            state.router_hot_cache_replicated_selected_expert_slots = 0;
        }

        struct RuntimeScratchBindings
        {
            int32_t *route_expert_ids = nullptr;
            float *route_weights = nullptr;
            int32_t *route_participant_ids = nullptr;
            int32_t *expert_counts = nullptr;
            int32_t *expert_offsets = nullptr;
            int32_t *grouped_token_ids = nullptr;
            float *grouped_route_weights = nullptr;
            float *grouped_gate_scratch = nullptr;
            float *grouped_up_scratch = nullptr;
            float *grouped_output_partials = nullptr;
            void *decode_scratch = nullptr;
            void *reserved_ptrs[3] = {};
            uint64_t reserved_u64[2] = {};
            uint32_t prefill_token_capacity = 0;
            uint32_t prefill_route_capacity = 0;
        };

        RuntimeScratchBindings captureRuntimeScratchBindings(const DeviceMoELayerRuntime &state) noexcept
        {
            RuntimeScratchBindings scratch;
            scratch.route_expert_ids = state.route_expert_ids;
            scratch.route_weights = state.route_weights;
            scratch.route_participant_ids = state.route_participant_ids;
            scratch.expert_counts = state.expert_counts;
            scratch.expert_offsets = state.expert_offsets;
            scratch.grouped_token_ids = state.grouped_token_ids;
            scratch.grouped_route_weights = state.grouped_route_weights;
            scratch.grouped_gate_scratch = state.grouped_gate_scratch;
            scratch.grouped_up_scratch = state.grouped_up_scratch;
            scratch.grouped_output_partials = state.grouped_output_partials;
            scratch.decode_scratch = state.decode_scratch;
            scratch.reserved_ptrs[0] = state.reserved_ptrs[0];
            scratch.reserved_ptrs[1] = state.reserved_ptrs[1];
            scratch.reserved_ptrs[2] = state.reserved_ptrs[2];
            scratch.reserved_u64[0] = state.reserved_u64[0];
            scratch.reserved_u64[1] = state.reserved_u64[1];
            scratch.prefill_token_capacity = state.prefill_token_capacity;
            scratch.prefill_route_capacity = state.prefill_route_capacity;
            return scratch;
        }

        void restoreRuntimeScratchBindings(DeviceMoELayerRuntime &state,
                                           const RuntimeScratchBindings &scratch) noexcept
        {
            state.route_expert_ids = scratch.route_expert_ids;
            state.route_weights = scratch.route_weights;
            state.route_participant_ids = scratch.route_participant_ids;
            state.expert_counts = scratch.expert_counts;
            state.expert_offsets = scratch.expert_offsets;
            state.grouped_token_ids = scratch.grouped_token_ids;
            state.grouped_route_weights = scratch.grouped_route_weights;
            state.grouped_gate_scratch = scratch.grouped_gate_scratch;
            state.grouped_up_scratch = scratch.grouped_up_scratch;
            state.grouped_output_partials = scratch.grouped_output_partials;
            state.decode_scratch = scratch.decode_scratch;
            state.reserved_ptrs[0] = scratch.reserved_ptrs[0];
            state.reserved_ptrs[1] = scratch.reserved_ptrs[1];
            state.reserved_ptrs[2] = scratch.reserved_ptrs[2];
            state.reserved_u64[0] = scratch.reserved_u64[0];
            state.reserved_u64[1] = scratch.reserved_u64[1];
            state.reserved_u64[2] = 0;
            state.reserved_u64[3] = 0;
            state.prefill_token_capacity = scratch.prefill_token_capacity;
            state.prefill_route_capacity = scratch.prefill_route_capacity;
        }

        void resetPerRequestRuntimeFields(DeviceMoELayerRuntime &state, int num_experts) noexcept
        {
            std::fill(state.decode_histogram, state.decode_histogram + num_experts, 0ULL);
            std::fill(state.decode_local_histogram, state.decode_local_histogram + num_experts, 0ULL);
            resetRouterHotCacheCounters(state);
            state.reserved_u64[2] = 0;
            state.reserved_u64[3] = 0;
        }

        /**
         * @brief Find a payload-ready descriptor in one placement bank.
         *
         * Portable prefix-cache state records the stable local slot that owned a
         * local expert at capture time.  When that slot id is known, restore must
         * not accept another ready descriptor for the same logical expert: doing
         * so can bind old transfer-slot pointers after the graph has rebuilt its
         * transfer-slot directory for the restored request.
         */
        bool findReadyDescriptorForExpertInBank(const DeviceMoEPlacementBank &bank,
                                                uint32_t expert,
                                                int32_t expected_local_slot,
                                                DeviceMoEExpertDescriptor &out) noexcept
        {
            if (expert >= bank.expert_count || expert >= kDeviceMoEMaxExperts)
                return false;
            const auto &candidate = bank.experts[expert];
            if (candidate.logical_expert_id != static_cast<int32_t>(expert) ||
                (expected_local_slot >= 0 && candidate.local_slot != expected_local_slot) ||
                !descriptorReady(candidate))
            {
                return false;
            }
            out = candidate;
            return true;
        }

        bool findReadyDescriptorForExpert(const DeviceMoELayerRuntime &state,
                                          uint32_t expert,
                                          int32_t expected_local_slot,
                                          DeviceMoEExpertDescriptor &out) noexcept
        {
            if (state.active_bank <= 1u &&
                findReadyDescriptorForExpertInBank(
                    state.banks[state.active_bank], expert, expected_local_slot, out))
            {
                return true;
            }
            for (uint32_t bank_idx = 0; bank_idx < 2u; ++bank_idx)
            {
                if (bank_idx == state.active_bank)
                    continue;
                if (findReadyDescriptorForExpertInBank(
                        state.banks[bank_idx], expert, expected_local_slot, out))
                {
                    return true;
                }
            }
            return false;
        }

        /**
         * @brief Validate a local payload descriptor returned during portable restore.
         *
         * The restore path will add placement flags such as LocalCompute after it
         * binds the payload descriptor, so this check intentionally focuses on the
         * pointer-bearing contract: the descriptor must name the requested expert,
         * have complete gate/up/down payloads, and match the stable local slot
         * recorded in the portable blob when that slot is known.
         */
        bool descriptorMatchesPortableLocalClaim(const DeviceMoEExpertDescriptor &desc,
                                                 int expert,
                                                 int32_t expected_local_slot) noexcept
        {
            return desc.logical_expert_id == static_cast<int32_t>(expert) &&
                   (expected_local_slot < 0 || desc.local_slot == expected_local_slot) &&
                   descriptorReady(desc);
        }

        /**
         * @brief Decide whether a portable local expert must be rebound by resolver.
         *
         * A local-compute expert whose owner is another participant represents a
         * graph-owned transfer-slot replica.  Prefix restore must ask the graph
         * side transfer-slot directory for the current descriptor before looking
         * at old runtime banks, because the old banks may still contain descriptor
         * shapes that are "ready" but point into stale device allocations.
         */
        bool portableLocalReplicaRequiresResolver(
            const DeviceMoEPortableLayerRuntimeState &snapshot,
            const DeviceMoEPortableExpertRuntimeState &saved,
            const DeviceMoERuntimeTable::LocalPayloadDescriptorResolver &resolver) noexcept
        {
            return resolver &&
                   saved.local_compute != 0u &&
                   saved.local_slot >= 0 &&
                   saved.owner_participant >= 0 &&
                   saved.owner_participant != static_cast<int32_t>(snapshot.participant_id);
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
        initial_host_layers_.resize(host_layers_.size());
        initial_layer_captured_.assign(host_layers_.size(), 0u);

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
               state.route_participant_ids &&
               state.expert_counts &&
               state.expert_offsets &&
               state.grouped_token_ids &&
               state.grouped_route_weights &&
               state.reserved_ptrs[0] &&
               state.reserved_ptrs[1] &&
               state.reserved_ptrs[2] &&
               state.reserved_u64[0] >=
                   static_cast<uint64_t>(num_experts_) * static_cast<uint64_t>(kDeviceMoEMaxParticipants) &&
               state.reserved_u64[1] >=
                   static_cast<uint64_t>(num_experts_) * static_cast<uint64_t>(kDeviceMoEMaxParticipants);
    }

    void DeviceMoERuntimeTable::recordDecodeHistogramProducerStream(void *stream)
    {
        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram producer stream must be explicit");
        }
        decode_histogram_producer_stream_ = stream;
    }

    void *DeviceMoERuntimeTable::decodeHistogramProducerStream() const
    {
        return decode_histogram_producer_stream_;
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

        if (mirror_to_device_ && !stream)
        {
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram sync requires an explicit stream");
        }

        std::vector<uint64_t> counts(static_cast<size_t>(num_layers_) * static_cast<size_t>(num_experts_), 0);
        std::vector<uint64_t> local_counts(static_cast<size_t>(num_layers_) * static_cast<size_t>(num_experts_), 0);

        if (!mirror_to_device_)
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                auto *dst = counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                std::copy(state.decode_histogram,
                          state.decode_histogram + num_experts_,
                          dst);
                auto *local_dst = local_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                std::copy(state.decode_local_histogram,
                          state.decode_local_histogram + num_experts_,
                          local_dst);
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

                const auto *local_src = device_layers_[layer_idx].decode_local_histogram;
                auto *local_dst = local_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                copyMirrorToHost(device_id_, local_dst, local_src,
                                 static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                                 stream,
                                 layerPrefix(layer_idx) + "decode local histogram D2H");
            }
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto *layer_counts = counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            histogram.mergeLayerCounts(layer_idx, layer_counts, num_experts_, /*count_window_tokens=*/false);

            if (PerfStatsCollector::isEnabled())
            {
                const auto *layer_local_counts =
                    local_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                const uint64_t selected_slots =
                    std::accumulate(layer_counts, layer_counts + num_experts_, uint64_t{0});
                const uint64_t local_slots =
                    std::accumulate(layer_local_counts, layer_local_counts + num_experts_, uint64_t{0});
                const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                const PerfStatsCollector::Tags tags{
                    {"layer", std::to_string(layer_idx)},
                    {"participant", std::to_string(state.participant_id)},
                    {"participants", std::to_string(state.participant_count)},
                    {"active_epoch", std::to_string(state.active_epoch)},
                    {"reset", reset_runtime_counts ? "true" : "false"}};
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "runtime_selected_slots",
                    static_cast<double>(selected_slots),
                    "rebalance",
                    device_id_.toString(),
                    tags);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "runtime_local_compute_slots",
                    static_cast<double>(local_slots),
                    "rebalance",
                    device_id_.toString(),
                    tags);
            }
        }

        if (!reset_runtime_counts)
            return true;

        for (auto &state : host_layers_)
        {
            std::fill(state.decode_histogram, state.decode_histogram + num_experts_, 0ULL);
            std::fill(state.decode_local_histogram, state.decode_local_histogram + num_experts_, 0ULL);
            resetRouterHotCacheCounters(state);
        }

        if (mirror_to_device_)
        {
            const size_t counters_offset = offsetof(DeviceMoELayerRuntime, router_hot_cache_eligible_dispatches);
            const size_t counters_bytes =
                offsetof(DeviceMoELayerRuntime, route_expert_ids) - counters_offset;
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                auto *dst = device_layers_[layer_idx].decode_histogram;
                memsetMirror(device_id_, dst, 0,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode histogram reset");
                auto *local_dst = device_layers_[layer_idx].decode_local_histogram;
                memsetMirror(device_id_, local_dst, 0,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode local histogram reset");
                auto *counter_dst =
                    reinterpret_cast<std::byte *>(device_layers_ + layer_idx) + counters_offset;
                memsetMirror(device_id_, counter_dst, 0,
                             counters_bytes,
                             stream,
                             layerPrefix(layer_idx) + "router hot-cache counter reset");
            }
            synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode histogram reset sync");
        }

        return true;
    }

    bool DeviceMoERuntimeTable::captureDecodeHistogramCounts(
        std::vector<uint64_t> &selected_counts,
        std::vector<uint64_t> &local_counts,
        void *stream)
    {
        const size_t entry_count =
            static_cast<size_t>(num_layers_) * static_cast<size_t>(num_experts_);
        selected_counts.assign(entry_count, 0ULL);
        local_counts.assign(entry_count, 0ULL);

        if (!mirror_to_device_)
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                const auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                auto *selected_dst =
                    selected_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                auto *local_dst =
                    local_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
                std::copy(state.decode_histogram,
                          state.decode_histogram + num_experts_,
                          selected_dst);
                std::copy(state.decode_local_histogram,
                          state.decode_local_histogram + num_experts_,
                          local_dst);
            }
            return true;
        }

        if (!stream)
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram capture requires an explicit stream");

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto *selected_src = device_layers_[layer_idx].decode_histogram;
            auto *selected_dst =
                selected_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            copyMirrorToHost(device_id_, selected_dst, selected_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode histogram capture");

            const auto *local_src = device_layers_[layer_idx].decode_local_histogram;
            auto *local_dst =
                local_counts.data() + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            copyMirrorToHost(device_id_, local_dst, local_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode local histogram capture");
        }
        synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode histogram capture sync");
        return true;
    }

    bool DeviceMoERuntimeTable::restoreDecodeHistogramCounts(
        const uint64_t *selected_counts,
        const uint64_t *local_counts,
        size_t layer_count,
        size_t expert_count,
        void *stream)
    {
        if (!selected_counts || !local_counts)
            return false;
        if (layer_count != static_cast<size_t>(num_layers_) ||
            expert_count != static_cast<size_t>(num_experts_))
        {
            LOG_ERROR("[MoERuntimeTable] decode histogram restore shape mismatch: table layers="
                      << num_layers_ << " experts=" << num_experts_
                      << " blob layers=" << layer_count
                      << " experts=" << expert_count);
            return false;
        }

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            const auto *selected_src =
                selected_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            const auto *local_src =
                local_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            std::copy(selected_src, selected_src + num_experts_, state.decode_histogram);
            std::copy(local_src, local_src + num_experts_, state.decode_local_histogram);
            resetRouterHotCacheCounters(state);
        }

        if (!mirror_to_device_)
            return true;

        if (!stream)
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored decode histogram restore requires an explicit stream");

        const size_t counters_offset = offsetof(DeviceMoELayerRuntime, router_hot_cache_eligible_dispatches);
        const size_t counters_bytes =
            offsetof(DeviceMoELayerRuntime, route_expert_ids) - counters_offset;
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto *selected_src =
                selected_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            auto *selected_dst = device_layers_[layer_idx].decode_histogram;
            copyHostToMirror(device_id_, selected_dst, selected_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode histogram restore");

            const auto *local_src =
                local_counts + static_cast<size_t>(layer_idx) * static_cast<size_t>(num_experts_);
            auto *local_dst = device_layers_[layer_idx].decode_local_histogram;
            copyHostToMirror(device_id_, local_dst, local_src,
                             static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                             stream,
                             layerPrefix(layer_idx) + "decode local histogram restore");

            auto *counter_dst =
                reinterpret_cast<std::byte *>(device_layers_ + layer_idx) + counters_offset;
            memsetMirror(device_id_, counter_dst, 0,
                         counters_bytes,
                         stream,
                         layerPrefix(layer_idx) + "router hot-cache counter restore reset");
        }
        return true;
    }

    void DeviceMoERuntimeTable::resetDecodeHistogramCounts(void *stream)
    {
        for (auto &state : host_layers_)
        {
            std::fill(state.decode_histogram, state.decode_histogram + num_experts_, 0ULL);
            std::fill(state.decode_local_histogram, state.decode_local_histogram + num_experts_, 0ULL);
            resetRouterHotCacheCounters(state);
        }

        if (!mirror_to_device_)
            return;

        const size_t counters_offset = offsetof(DeviceMoELayerRuntime, router_hot_cache_eligible_dispatches);
        const size_t counters_bytes =
            offsetof(DeviceMoELayerRuntime, route_expert_ids) - counters_offset;
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            auto *dst = device_layers_[layer_idx].decode_histogram;
            memsetMirror(device_id_, dst, 0,
                         static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                         stream,
                         layerPrefix(layer_idx) + "decode histogram reset");
            auto *local_dst = device_layers_[layer_idx].decode_local_histogram;
            memsetMirror(device_id_, local_dst, 0,
                         static_cast<size_t>(num_experts_) * sizeof(uint64_t),
                         stream,
                         layerPrefix(layer_idx) + "decode local histogram reset");
            auto *counter_dst =
                reinterpret_cast<std::byte *>(device_layers_ + layer_idx) + counters_offset;
            memsetMirror(device_id_, counter_dst, 0,
                         counters_bytes,
                         stream,
                         layerPrefix(layer_idx) + "router hot-cache counter reset");
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
            state.route_participant_ids = scratch.route_participant_ids;
            state.expert_counts = scratch.expert_counts;
            state.expert_offsets = scratch.expert_offsets;
            state.grouped_token_ids = scratch.grouped_token_ids;
            state.grouped_route_weights = scratch.grouped_route_weights;
            state.reserved_ptrs[0] = scratch.llep_split_ends;
            state.reserved_ptrs[1] = scratch.llep_assignment_spans;
            state.reserved_ptrs[2] = scratch.llep_weight_transfers;
            state.reserved_u64[0] = scratch.llep_plan_capacity;
            state.reserved_u64[1] = scratch.llep_plan_capacity;
            state.reserved_u64[2] = 0;
            state.reserved_u64[3] = 0;
            state.prefill_token_capacity = scratch.token_capacity;
            state.prefill_route_capacity = scratch.route_capacity;
        }

        if (!mirror_to_device_)
            return;

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            uploadLayerState(layer_idx, stream);
        synchronizeMirror(device_id_, stream, "[MoERuntimeTable] decode runtime reset sync");
    }

    bool DeviceMoERuntimeTable::hasInitialRuntimeState() const noexcept
    {
        return std::any_of(initial_layer_captured_.begin(),
                           initial_layer_captured_.end(),
                           [](uint8_t captured)
                           { return captured != 0u; });
    }

    void DeviceMoERuntimeTable::restoreInitialRuntimeState(void *stream)
    {
        void *owned_stream = nullptr;
        void *active_stream = stream;
        if (mirror_to_device_ && !active_stream)
        {
            owned_stream = createMirrorStream(device_id_,
                                              "[MoERuntimeTable] initial runtime restore stream");
            active_stream = owned_stream;
        }

        try
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                const auto scratch = captureRuntimeScratchBindings(state);

                if (initial_layer_captured_[static_cast<size_t>(layer_idx)] != 0u)
                    state = initial_host_layers_[static_cast<size_t>(layer_idx)];
                else
                    resetLayer(state);

                restoreRuntimeScratchBindings(state, scratch);
                resetPerRequestRuntimeFields(state, num_experts_);

                if (mirror_to_device_)
                    uploadLayerState(layer_idx, active_stream);
            }

            if (mirror_to_device_)
                synchronizeMirror(device_id_, active_stream,
                                  "[MoERuntimeTable] initial runtime restore sync");

            if (owned_stream)
            {
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] initial runtime restore stream destroy");
                owned_stream = nullptr;
            }
        }
        catch (...)
        {
            if (owned_stream)
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] initial runtime restore stream destroy");
            throw;
        }
    }

    void DeviceMoERuntimeTable::syncRuntimeStateToHost(void *stream)
    {
        if (!mirror_to_device_)
            return;

        void *owned_stream = nullptr;
        void *active_stream = stream;
        if (!active_stream)
        {
            owned_stream = createMirrorStream(device_id_,
                                              "[MoERuntimeTable] runtime state D2H stream");
            active_stream = owned_stream;
        }

        try
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                copyMirrorToHost(device_id_,
                                 host_layers_.data() + layer_idx,
                                 device_layers_ + layer_idx,
                                 sizeof(DeviceMoELayerRuntime),
                                 active_stream,
                                 layerPrefix(layer_idx) + "runtime table D2H");
            }
            synchronizeMirror(device_id_, active_stream,
                              "[MoERuntimeTable] runtime state D2H sync");

            if (owned_stream)
            {
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime state D2H stream destroy");
                owned_stream = nullptr;
            }
        }
        catch (...)
        {
            if (owned_stream)
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime state D2H stream destroy");
            throw;
        }
    }

    void DeviceMoERuntimeTable::restoreRuntimeStateSnapshot(
        const DeviceMoELayerRuntime *layers,
        size_t layer_count,
        void *stream)
    {
        if (!layers)
            throw std::invalid_argument("[MoERuntimeTable] runtime snapshot restore requires layer data");
        if (layer_count != static_cast<size_t>(num_layers_))
            throw std::invalid_argument("[MoERuntimeTable] runtime snapshot layer count mismatch");

        void *owned_stream = nullptr;
        void *active_stream = stream;
        if (mirror_to_device_ && !active_stream)
        {
            owned_stream = createMirrorStream(device_id_,
                                              "[MoERuntimeTable] runtime snapshot restore stream");
            active_stream = owned_stream;
        }

        try
        {
            for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
            {
                auto &state = host_layers_[static_cast<size_t>(layer_idx)];
                const auto scratch = captureRuntimeScratchBindings(state);
                const auto &snapshot = layers[static_cast<size_t>(layer_idx)];
                if (snapshot.expert_count != static_cast<uint32_t>(num_experts_) ||
                    snapshot.top_k != static_cast<uint32_t>(top_k_) ||
                    snapshot.active_bank > 1u)
                {
                    throw std::invalid_argument(
                        layerPrefix(layer_idx) + "runtime snapshot metadata mismatch");
                }

                if (snapshot.active_epoch == 0u)
                {
                    resetPerRequestRuntimeFields(state, num_experts_);
                    if (mirror_to_device_)
                        uploadLayerState(layer_idx, active_stream);
                    continue;
                }

                state = snapshot;
                restoreRuntimeScratchBindings(state, scratch);
                resetPerRequestRuntimeFields(state, num_experts_);
                if (mirror_to_device_)
                    uploadLayerState(layer_idx, active_stream);
            }

            if (mirror_to_device_)
                synchronizeMirror(device_id_, active_stream,
                                  "[MoERuntimeTable] runtime snapshot restore sync");

            if (owned_stream)
            {
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime snapshot restore stream destroy");
                owned_stream = nullptr;
            }
        }
        catch (...)
        {
            if (owned_stream)
                destroyMirrorStream(device_id_, owned_stream,
                                    "[MoERuntimeTable] runtime snapshot restore stream destroy");
            throw;
        }
    }

    bool DeviceMoERuntimeTable::capturePortableRuntimeState(
        std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
        void *stream)
    {
        layers.clear();
        const DeviceMoELayerRuntime *source_layers = host_layers_.data();
        std::vector<DeviceMoELayerRuntime> mirrored_layers;

        if (mirror_to_device_)
        {
            if (!stream)
                throw std::invalid_argument(
                    "[MoERuntimeTable] mirrored portable runtime capture requires an explicit stream");
            mirrored_layers.resize(static_cast<size_t>(num_layers_));
            copyMirrorToHost(device_id_,
                             mirrored_layers.data(),
                             device_layers_,
                             sizeof(DeviceMoELayerRuntime) *
                                 static_cast<size_t>(num_layers_),
                             stream,
                             "[MoERuntimeTable] portable runtime state capture");
            synchronizeMirror(device_id_, stream,
                              "[MoERuntimeTable] portable runtime state capture sync");
            source_layers = mirrored_layers.data();
        }

        layers.reserve(static_cast<size_t>(num_layers_));
        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto &state = source_layers[static_cast<size_t>(layer_idx)];
            if (state.expert_count != static_cast<uint32_t>(num_experts_) ||
                state.top_k != static_cast<uint32_t>(top_k_) ||
                state.active_bank > 1u)
            {
                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                     << ": cannot capture malformed portable runtime state"
                                                     << " active_bank=" << state.active_bank
                                                     << " experts=" << state.expert_count
                                                     << " top_k=" << state.top_k);
                layers.clear();
                return false;
            }

            const auto &bank = state.banks[state.active_bank];
            DeviceMoEPortableLayerRuntimeState captured;
            captured.active_epoch = state.active_epoch;
            captured.expert_count = state.expert_count;
            captured.top_k = state.top_k;
            captured.participant_id = state.participant_id;
            captured.participant_count = state.participant_count;
            captured.experts.resize(static_cast<size_t>(num_experts_));
            captured.selected_histogram.assign(
                state.decode_histogram,
                state.decode_histogram + num_experts_);
            captured.local_histogram.assign(
                state.decode_local_histogram,
                state.decode_local_histogram + num_experts_);

            for (int expert = 0; expert < num_experts_; ++expert)
            {
                const auto &desc = bank.experts[static_cast<size_t>(expert)];
                auto &dst = captured.experts[static_cast<size_t>(expert)];
                dst.logical_expert_id =
                    desc.logical_expert_id >= 0 ? desc.logical_expert_id : expert;
                dst.owner_participant = desc.owner_participant;
                dst.local_slot = desc.local_slot;
                dst.flags = portableMoEExpertFlags(desc.flags);
                dst.local_compute = bank.local_compute_mask[static_cast<size_t>(expert)] != 0u ? 1u : 0u;
                dst.replica_role = bank.replica_role[static_cast<size_t>(expert)];
                dst.resident_participant_mask =
                    bank.resident_participant_mask[static_cast<size_t>(expert)];
            }
            layers.push_back(std::move(captured));
        }

        return true;
    }

    bool DeviceMoERuntimeTable::restorePortableRuntimeState(
        const std::vector<DeviceMoEPortableLayerRuntimeState> &layers,
        void *stream,
        const LocalPayloadDescriptorResolver &local_payload_resolver)
    {
        if (layers.size() != static_cast<size_t>(num_layers_))
        {
            LOG_ERROR("[MoERuntimeTable] portable runtime restore layer count mismatch: table="
                      << num_layers_ << " snapshot=" << layers.size());
            return false;
        }
        if (mirror_to_device_ && !stream)
            throw std::invalid_argument(
                "[MoERuntimeTable] mirrored portable runtime restore requires an explicit stream");

        for (int layer_idx = 0; layer_idx < num_layers_; ++layer_idx)
        {
            const auto &snapshot = layers[static_cast<size_t>(layer_idx)];
            if (snapshot.expert_count != static_cast<uint32_t>(num_experts_) ||
                snapshot.top_k != static_cast<uint32_t>(top_k_) ||
                snapshot.experts.size() != static_cast<size_t>(num_experts_) ||
                snapshot.selected_histogram.size() != static_cast<size_t>(num_experts_) ||
                snapshot.local_histogram.size() != static_cast<size_t>(num_experts_))
            {
                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                     << ": portable runtime restore metadata mismatch");
                return false;
            }

            auto &state = host_layers_[static_cast<size_t>(layer_idx)];
            MoEPlacementUpdate update;
            update.epoch = std::max<uint32_t>(
                state.active_epoch + 1u,
                snapshot.active_epoch == 0u ? 1u : snapshot.active_epoch);
            if (update.epoch <= state.active_epoch)
                update.epoch = state.active_epoch + 1u;
            update.expert_count = static_cast<uint32_t>(num_experts_);
            update.participant_id = snapshot.participant_id;
            update.participant_count = snapshot.participant_count;
            update.experts.resize(static_cast<size_t>(num_experts_));
            update.local_compute_mask.assign(static_cast<size_t>(num_experts_), 0u);
            update.replica_role.assign(
                static_cast<size_t>(num_experts_),
                static_cast<uint8_t>(DeviceMoEReplicaRole::None));
            update.resident_participant_mask.assign(static_cast<size_t>(num_experts_), 0u);

            for (int expert = 0; expert < num_experts_; ++expert)
            {
                const auto &saved = snapshot.experts[static_cast<size_t>(expert)];
                if (saved.logical_expert_id != static_cast<int32_t>(expert))
                {
                    LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                         << ": portable runtime restore logical expert mismatch"
                                                         << " slot=" << expert
                                                         << " logical=" << saved.logical_expert_id);
                    return false;
                }

                DeviceMoEExpertDescriptor desc;
                desc.logical_expert_id = expert;
                desc.owner_participant = saved.owner_participant;
                desc.local_slot = saved.local_compute ? saved.local_slot : -1;
                desc.flags = clearPayloadBearingMoEExpertFlags(
                    portableMoEExpertFlags(saved.flags));

                if (saved.local_compute != 0u)
                {
                    DeviceMoEExpertDescriptor ready_desc;
                    bool has_ready_descriptor = false;
                    const bool resolver_is_authoritative =
                        portableLocalReplicaRequiresResolver(snapshot,
                                                             saved,
                                                             local_payload_resolver);

                    if (resolver_is_authoritative)
                    {
                        has_ready_descriptor =
                            local_payload_resolver(layer_idx,
                                                   expert,
                                                   saved.local_slot,
                                                   ready_desc);
                        if (!has_ready_descriptor)
                        {
                            LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                                 << ": portable runtime restore requires graph-owned local payload for remote-owned expert "
                                                                 << expert
                                                                 << " slot=" << saved.local_slot
                                                                 << " but the resolver could not bind it");
                            return false;
                        }
                        if (!descriptorMatchesPortableLocalClaim(ready_desc,
                                                                 expert,
                                                                 saved.local_slot))
                        {
                            LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                                 << ": portable runtime restore resolver returned an invalid descriptor for expert "
                                                                 << expert);
                            return false;
                        }
                    }
                    else
                    {
                        has_ready_descriptor =
                            findReadyDescriptorForExpert(state,
                                                         static_cast<uint32_t>(expert),
                                                         saved.local_slot,
                                                         ready_desc);
                        if (!has_ready_descriptor && local_payload_resolver)
                        {
                            has_ready_descriptor =
                                local_payload_resolver(layer_idx,
                                                       expert,
                                                       saved.local_slot,
                                                       ready_desc);
                            if (has_ready_descriptor &&
                                !descriptorMatchesPortableLocalClaim(ready_desc,
                                                                     expert,
                                                                     saved.local_slot))
                            {
                                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                                     << ": portable runtime restore resolver returned an invalid descriptor for expert "
                                                                     << expert);
                                return false;
                            }
                        }
                    }

                    if (!has_ready_descriptor)
                    {
                        LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                             << ": portable runtime restore requires local payload for expert "
                                                             << expert
                                                             << " but no live descriptor is resident");
                        return false;
                    }
                    desc = ready_desc;
                    desc.logical_expert_id = expert;
                    desc.owner_participant = saved.owner_participant;
                    if (saved.local_slot >= 0)
                        desc.local_slot = saved.local_slot;
                    desc.flags = portableMoEExpertFlags(saved.flags) |
                                 toMoEExpertFlags(DeviceMoEExpertFlags::Valid) |
                                 toMoEExpertFlags(DeviceMoEExpertFlags::Resident) |
                                 toMoEExpertFlags(DeviceMoEExpertFlags::LocalCompute);
                    if (hasMoEExpertFlag(ready_desc.flags,
                                         DeviceMoEExpertFlags::TransferSlot))
                    {
                        desc.flags |= toMoEExpertFlags(DeviceMoEExpertFlags::TransferSlot);
                    }
                }

                update.experts[static_cast<size_t>(expert)] = desc;
                update.local_compute_mask[static_cast<size_t>(expert)] =
                    saved.local_compute != 0u ? 1u : 0u;
                update.replica_role[static_cast<size_t>(expert)] = saved.replica_role;
                update.resident_participant_mask[static_cast<size_t>(expert)] =
                    saved.resident_participant_mask;
            }

            try
            {
                prepareInactiveBank(layer_idx, update);
                flipActiveBank(layer_idx, update.epoch, stream);
            }
            catch (const std::exception &ex)
            {
                LOG_ERROR("[MoERuntimeTable] layer " << layer_idx
                                                     << ": portable runtime restore failed: "
                                                     << ex.what());
                return false;
            }

            auto &restored = host_layers_[static_cast<size_t>(layer_idx)];
            std::copy(snapshot.selected_histogram.begin(),
                      snapshot.selected_histogram.end(),
                      restored.decode_histogram);
            std::copy(snapshot.local_histogram.begin(),
                      snapshot.local_histogram.end(),
                      restored.decode_local_histogram);
            resetRouterHotCacheCounters(restored);
            restored.reserved_u64[2] = 0;
            restored.reserved_u64[3] = 0;
            if (mirror_to_device_)
                uploadLayerState(layer_idx, stream);
        }

        return true;
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
        state.participant_id = update.participant_id;
        state.participant_count = update.participant_count;
        bank = {};
        bank.epoch = update.epoch;
        bank.expert_count = update.expert_count;

        for (uint32_t expert = 0; expert < update.expert_count; ++expert)
        {
            bank.experts[expert] = update.experts[expert];
            bank.local_compute_mask[expert] = update.local_compute_mask[expert];
            bank.replica_role[expert] = update.replica_role[expert];
            const uint32_t resident_mask = residentParticipantMaskForUpdate(update, expert);
            bank.resident_participant_mask[expert] = resident_mask;
            if (participantMaskCount(resident_mask) > 1u)
                ++bank.reserved[0];
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
        captureInitialLayerStateIfNeeded(layer_idx);

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
        if (update.participant_count == 0 || update.participant_count > kDeviceMoEMaxParticipants)
            throw std::invalid_argument(layerPrefix(layer_idx) + "participant_count must be in [1, " +
                                        std::to_string(kDeviceMoEMaxParticipants) + "]");
        if (update.participant_id >= update.participant_count)
            throw std::invalid_argument(layerPrefix(layer_idx) + "participant_id must be less than participant_count");
        if (update.experts.size() != update.expert_count ||
            update.local_compute_mask.size() != update.expert_count ||
            update.replica_role.size() != update.expert_count ||
            (!update.resident_participant_mask.empty() &&
             update.resident_participant_mask.size() != update.expert_count))
        {
            throw std::invalid_argument(layerPrefix(layer_idx) + "placement update vectors must match expert_count");
        }

        const uint32_t valid_participant_mask = participantMaskLimit(update.participant_count);
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

            const uint32_t resident_mask = residentParticipantMaskForUpdate(update, expert);
            if ((resident_mask & ~valid_participant_mask) != 0u)
                throw std::invalid_argument(layerPrefix(layer_idx) + "resident_participant_mask names a participant outside participant_count");
            if (update.local_compute_mask[expert] != 0u &&
                (resident_mask & participantBit(update.participant_id)) == 0u)
            {
                throw std::invalid_argument(layerPrefix(layer_idx) + "resident_participant_mask must include local participant for local compute");
            }
            if (desc.owner_participant >= 0 &&
                desc.owner_participant < static_cast<int32_t>(update.participant_count) &&
                (resident_mask & participantBit(static_cast<uint32_t>(desc.owner_participant))) == 0u)
            {
                throw std::invalid_argument(layerPrefix(layer_idx) + "resident_participant_mask must include the owner participant");
            }

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
        state.participant_id = 0;
        state.participant_count = 1;
        state.banks[0].expert_count = static_cast<uint32_t>(num_experts_);
        state.banks[1].expert_count = static_cast<uint32_t>(num_experts_);
    }

    void DeviceMoERuntimeTable::captureInitialLayerStateIfNeeded(int layer_idx)
    {
        const auto idx = static_cast<size_t>(layer_idx);
        if (idx >= initial_layer_captured_.size() ||
            initial_layer_captured_[idx] != 0u)
        {
            return;
        }

        initial_host_layers_[idx] = host_layers_[idx];
        resetPerRequestRuntimeFields(initial_host_layers_[idx], num_experts_);
        initial_layer_captured_[idx] = 1u;
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
               allocation.route_participant_ids &&
               allocation.expert_counts &&
               allocation.expert_offsets &&
               allocation.grouped_token_ids &&
               allocation.grouped_route_weights &&
               allocation.llep_split_ends &&
               allocation.llep_assignment_spans &&
               allocation.llep_weight_transfers &&
               allocation.llep_plan_capacity >=
                   static_cast<uint32_t>(num_experts_) * kDeviceMoEMaxParticipants;
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
            if (scratch.route_participant_ids)
                freeMirror(device_id_, scratch.route_participant_ids, layerPrefix(layer_idx) + "free prefill route_participant_ids");
            if (scratch.expert_counts)
                freeMirror(device_id_, scratch.expert_counts, layerPrefix(layer_idx) + "free prefill expert_counts");
            if (scratch.expert_offsets)
                freeMirror(device_id_, scratch.expert_offsets, layerPrefix(layer_idx) + "free prefill expert_offsets");
            if (scratch.grouped_token_ids)
                freeMirror(device_id_, scratch.grouped_token_ids, layerPrefix(layer_idx) + "free prefill grouped_token_ids");
            if (scratch.grouped_route_weights)
                freeMirror(device_id_, scratch.grouped_route_weights, layerPrefix(layer_idx) + "free prefill grouped_route_weights");
            if (scratch.llep_split_ends)
                freeMirror(device_id_, scratch.llep_split_ends, layerPrefix(layer_idx) + "free prefill llep_split_ends");
            if (scratch.llep_assignment_spans)
                freeMirror(device_id_, scratch.llep_assignment_spans, layerPrefix(layer_idx) + "free prefill llep_assignment_spans");
            if (scratch.llep_weight_transfers)
                freeMirror(device_id_, scratch.llep_weight_transfers, layerPrefix(layer_idx) + "free prefill llep_weight_transfers");
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
            const size_t llep_plan_capacity =
                static_cast<size_t>(num_experts_) * static_cast<size_t>(kDeviceMoEMaxParticipants);
            allocate(&allocation.route_expert_ids, route_capacity, "prefill route_expert_ids");
            allocate(&allocation.route_weights, route_capacity, "prefill route_weights");
            allocate(&allocation.route_participant_ids, route_capacity, "prefill route_participant_ids");
            allocate(&allocation.expert_counts, static_cast<size_t>(num_experts_), "prefill expert_counts");
            allocate(&allocation.expert_offsets, static_cast<size_t>(num_experts_), "prefill expert_offsets");
            allocate(&allocation.grouped_token_ids, route_capacity, "prefill grouped_token_ids");
            allocate(&allocation.grouped_route_weights, route_capacity, "prefill grouped_route_weights");
            allocate(&allocation.llep_split_ends,
                     static_cast<size_t>(num_experts_) * static_cast<size_t>(kDeviceMoEMaxParticipants),
                     "prefill llep_split_ends");
            allocate(&allocation.llep_assignment_spans,
                     llep_plan_capacity,
                     "prefill llep_assignment_spans");
            allocate(&allocation.llep_weight_transfers,
                     llep_plan_capacity,
                     "prefill llep_weight_transfers");
        }
        catch (...)
        {
            free_allocation(allocation);
            throw;
        }

        allocation.token_capacity = static_cast<uint32_t>(token_capacity);
        allocation.route_capacity = route_capacity;
        allocation.expert_capacity = static_cast<uint32_t>(num_experts_);
        allocation.llep_plan_capacity =
            static_cast<uint32_t>(static_cast<size_t>(num_experts_) *
                                  static_cast<size_t>(kDeviceMoEMaxParticipants));

        auto &state = host_layers_[static_cast<size_t>(layer_idx)];
        state.route_expert_ids = allocation.route_expert_ids;
        state.route_weights = allocation.route_weights;
        state.route_participant_ids = allocation.route_participant_ids;
        state.expert_counts = allocation.expert_counts;
        state.expert_offsets = allocation.expert_offsets;
        state.grouped_token_ids = allocation.grouped_token_ids;
        state.grouped_route_weights = allocation.grouped_route_weights;
        state.reserved_ptrs[0] = allocation.llep_split_ends;
        state.reserved_ptrs[1] = allocation.llep_assignment_spans;
        state.reserved_ptrs[2] = allocation.llep_weight_transfers;
        state.reserved_u64[0] = allocation.llep_plan_capacity;
        state.reserved_u64[1] = allocation.llep_plan_capacity;
        state.reserved_u64[2] = 0;
        state.reserved_u64[3] = 0;
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
            if (allocation.route_participant_ids)
                freeMirror(device_id_, allocation.route_participant_ids, "[MoERuntimeTable] free prefill route_participant_ids");
            if (allocation.expert_counts)
                freeMirror(device_id_, allocation.expert_counts, "[MoERuntimeTable] free prefill expert_counts");
            if (allocation.expert_offsets)
                freeMirror(device_id_, allocation.expert_offsets, "[MoERuntimeTable] free prefill expert_offsets");
            if (allocation.grouped_token_ids)
                freeMirror(device_id_, allocation.grouped_token_ids, "[MoERuntimeTable] free prefill grouped_token_ids");
            if (allocation.grouped_route_weights)
                freeMirror(device_id_, allocation.grouped_route_weights, "[MoERuntimeTable] free prefill grouped_route_weights");
            if (allocation.llep_split_ends)
                freeMirror(device_id_, allocation.llep_split_ends, "[MoERuntimeTable] free prefill llep_split_ends");
            if (allocation.llep_assignment_spans)
                freeMirror(device_id_, allocation.llep_assignment_spans, "[MoERuntimeTable] free prefill llep_assignment_spans");
            if (allocation.llep_weight_transfers)
                freeMirror(device_id_, allocation.llep_weight_transfers, "[MoERuntimeTable] free prefill llep_weight_transfers");
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
