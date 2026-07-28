/**
 * @file HybridGDNDeviceStateArena.h
 * @brief Cache-owned, preplanned GPU storage for every GDN layer state bank.
 *
 * The arena translates model-level GDN geometry into one contiguous
 * DeviceWorkspaceManager allocation.  Each hybrid-cache layer receives
 * isolated local/full/request slices, and later kernel construction binds
 * those slices through GDNDeviceStateBinding.  No kernel allocation or
 * graph-time capacity repair is permitted.
 */

#pragma once

#include "HybridKVCacheConfig.h"
#include "../backends/DeviceId.h"
#include "../execution/local_execution/device/DeviceWorkspaceManager.h"

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Own the persistent GDN state allocation for one hybrid GPU cache.
     *
     * This class is intentionally backend-neutral. DeviceWorkspaceManager
     * selects CUDA or ROCm from DeviceId, while the identical naming and
     * slicing policy keeps both GPU backends structurally symmetric.
     */
    class HybridGDNDeviceStateArena
    {
    public:
        HybridGDNDeviceStateArena() = default;
        ~HybridGDNDeviceStateArena() = default;

        HybridGDNDeviceStateArena(const HybridGDNDeviceStateArena &) = delete;
        HybridGDNDeviceStateArena &operator=(const HybridGDNDeviceStateArena &) = delete;
        HybridGDNDeviceStateArena(HybridGDNDeviceStateArena &&) = delete;
        HybridGDNDeviceStateArena &operator=(HybridGDNDeviceStateArena &&) = delete;

        /**
         * @brief Allocate and bind every layer's local/full/request state.
         *
         * @param device CUDA or ROCm device that owns the cache.
         * @param request_capacity Maximum independent request slots.
         * @param states Per-GDN-layer geometry and resulting bindings.
         * @param initialization_stream Explicit stream used to zero the arena.
         *
         * @throws std::runtime_error for malformed geometry, a null stream,
         *         allocation failure, zero-initialization failure, or any
         *         missing/undersized named slice.
         */
        void initialize(
            DeviceId device,
            int request_capacity,
            std::vector<HybridGDNLayerState> &states,
            void *initialization_stream)
        {
            if (!device.is_gpu())
                throw std::runtime_error(
                    "HybridGDNDeviceStateArena requires a CUDA or ROCm device");
            if (request_capacity <= 0)
                throw std::runtime_error(
                    "HybridGDNDeviceStateArena requires positive request capacity");
            if (!initialization_stream)
                throw std::runtime_error(
                    "HybridGDNDeviceStateArena requires an explicit initialization stream");
            if (states.empty())
                return;
            if (workspace_)
                throw std::runtime_error(
                    "HybridGDNDeviceStateArena cannot be initialized twice");

            WorkspaceRequirements requirements;
            for (size_t layer = 0; layer < states.size(); ++layer)
            {
                appendKernelRequirements(
                    requirements,
                    layer,
                    "conv",
                    states[layer].local_conv_state_size,
                    states[layer].full_conv_state_size,
                    request_capacity);
                appendKernelRequirements(
                    requirements,
                    layer,
                    "recurrence",
                    states[layer].local_recurrence_state_size,
                    states[layer].full_recurrence_state_size,
                    request_capacity);
            }

            if (requirements.buffers.empty())
                return;

            const size_t budget = requirements.total_bytes_with_alignment();
            workspace_ = std::make_unique<DeviceWorkspaceManager>(device, budget);
            if (!workspace_->allocate(requirements))
            {
                workspace_.reset();
                throw std::runtime_error(
                    "HybridGDNDeviceStateArena failed to allocate planned device state");
            }
            if (!workspace_->zeroAll(initialization_stream))
            {
                workspace_.reset();
                throw std::runtime_error(
                    "HybridGDNDeviceStateArena failed to initialize device state");
            }

            for (size_t layer = 0; layer < states.size(); ++layer)
            {
                states[layer].conv_device_state = resolveBinding(
                    layer,
                    "conv",
                    states[layer].local_conv_state_size,
                    states[layer].full_conv_state_size,
                    request_capacity);
                states[layer].recurrence_device_state = resolveBinding(
                    layer,
                    "recurrence",
                    states[layer].local_recurrence_state_size,
                    states[layer].full_recurrence_state_size,
                    request_capacity);
            }
        }

        /**
         * @brief Report bytes reserved for persistent GDN state.
         */
        size_t bytes() const noexcept
        {
            return workspace_ ? workspace_->used() : 0;
        }

    private:
        std::unique_ptr<DeviceWorkspaceManager> workspace_;

        static std::string bufferName(
            size_t layer,
            const char *kernel,
            const char *bank)
        {
            return "hybrid_gdn_layer_" + std::to_string(layer) +
                   "_" + kernel + "_" + bank;
        }

        static void appendKernelRequirements(
            WorkspaceRequirements &requirements,
            size_t layer,
            const char *kernel,
            int local_state_floats,
            int full_state_floats,
            int request_capacity)
        {
            if (local_state_floats <= 0)
                return;

            requirements.buffers.push_back({
                bufferName(layer, kernel, "primary"),
                static_cast<size_t>(local_state_floats) * sizeof(float),
                256,
                true});

            const int distinct_full_state_floats =
                full_state_floats > 0 ? full_state_floats : local_state_floats;
            if (distinct_full_state_floats != local_state_floats)
            {
                requirements.buffers.push_back({
                    bufferName(layer, kernel, "secondary"),
                    static_cast<size_t>(distinct_full_state_floats) * sizeof(float),
                    256,
                    true});
            }

            const int largest_state_floats =
                std::max(local_state_floats, distinct_full_state_floats);
            requirements.buffers.push_back({
                bufferName(layer, kernel, "requests"),
                static_cast<size_t>(request_capacity) *
                    static_cast<size_t>(largest_state_floats) * sizeof(float),
                256,
                true});
        }

        GDNDeviceStateBinding resolveBinding(
            size_t layer,
            const char *kernel,
            int local_state_floats,
            int full_state_floats,
            int request_capacity) const
        {
            if (local_state_floats <= 0)
                return {};

            GDNDeviceStateBinding binding;
            binding.primary_state = static_cast<float *>(
                workspace_->getBuffer(bufferName(layer, kernel, "primary")));
            binding.primary_state_floats = local_state_floats;

            const int effective_full_state_floats =
                full_state_floats > 0 ? full_state_floats : local_state_floats;
            if (effective_full_state_floats != local_state_floats)
            {
                binding.secondary_state = static_cast<float *>(
                    workspace_->getBuffer(bufferName(layer, kernel, "secondary")));
                binding.secondary_state_floats = effective_full_state_floats;
            }

            const std::string request_name =
                bufferName(layer, kernel, "requests");
            binding.request_state_bank = static_cast<float *>(
                workspace_->getBuffer(request_name));
            binding.request_state_bank_floats =
                workspace_->getBufferSize(request_name) / sizeof(float);
            binding.request_capacity = request_capacity;

            if (!binding.valid())
            {
                throw std::runtime_error(
                    "HybridGDNDeviceStateArena produced an invalid " +
                    std::string(kernel) + " binding for layer " +
                    std::to_string(layer));
            }
            return binding;
        }
    };
} // namespace llaminar2
