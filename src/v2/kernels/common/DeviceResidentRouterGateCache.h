/**
 * @file DeviceResidentRouterGateCache.h
 * @brief Stable device-owned identity for prepared MoE router gate weights.
 *
 * GPU graph builders may create a fresh `ITensor` view for the same immutable
 * model weight on every request. Those host wrapper addresses are not weight
 * identities and must never control persistent device-cache ownership. Router
 * gate conversions are instead identified by the workspace allocation
 * generation that owns the converted bytes, the source device allocation, and
 * the matrix geometry.
 */

#pragma once

#include <cstdint>

namespace llaminar2
{
    /**
     * @brief Device-resident identity for one immutable router gate matrix.
     *
     * `workspace_id` prevents an address reused by a replacement graph
     * workspace from matching stale converted storage. `source_device_ptr`
     * identifies the immutable model-weight allocation. Geometry completes the
     * identity and protects against an allocation being rebound with a
     * different matrix view.
     *
     * Host tensor objects and host data pointers are deliberately absent. A
     * request may rebuild those wrappers without changing the model weight, and
     * such a rebuild must remain a cache hit.
     */
    struct DeviceResidentRouterGateCacheKey
    {
        std::uint64_t workspace_id = 0;
        std::uintptr_t source_device_ptr = 0;
        int d_model = 0;
        int num_experts = 0;

        /**
         * @brief Construct a key from graph-owned device state.
         */
        static DeviceResidentRouterGateCacheKey make(
            std::uint64_t workspace_generation,
            const void *device_ptr,
            int model_width,
            int expert_count) noexcept
        {
            return {
                .workspace_id = workspace_generation,
                .source_device_ptr =
                    reinterpret_cast<std::uintptr_t>(device_ptr),
                .d_model = model_width,
                .num_experts = expert_count,
            };
        }

        /**
         * @brief Return true when both keys name the same prepared weight.
         */
        friend bool operator==(
            const DeviceResidentRouterGateCacheKey &left,
            const DeviceResidentRouterGateCacheKey &right) noexcept = default;
    };
} // namespace llaminar2
