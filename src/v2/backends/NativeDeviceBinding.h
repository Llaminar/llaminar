/**
 * @file NativeDeviceBinding.h
 * @brief Authenticate thread-local native device selection without a shadow cache.
 *
 * Current-device state can change through any runtime, library or collective
 * entrypoint. It is not an immutable hardware fact or a function-local remembered
 * ordinal. Query the actual thread state and select only when it differs; both
 * operations preserve native status and introduce no GPU synchronization.
 */
#pragma once
#include <functional>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    /** @brief Operation producing the retained native status. */
    enum class NativeDeviceBindingStep { CurrentDeviceQuery, DeviceSelection, AlreadySelected };
    /** @brief Exact native selection observation; failure never becomes a successful cache hit. */
    struct NativeDeviceBindingResult
    {
        NativeDeviceBindingStep step;
        int native_status;
        int observed_device;
        /** @return Whether the actual native operation succeeded. */
        bool succeeded() const noexcept { return native_status == 0; }
    };
    /**
     * @brief Bind one exact ordinal against current thread state, not remembered prior calls.
     * @param requested Exact nonnegative native ordinal.
     * @param current Native get-device operation or recorded device-free observation.
     * @param select Native set-device operation or recorded device-free selection.
     * @return Original native status and the operation that produced it.
     * @throws std::invalid_argument For a negative requested ordinal before native work.
     * @throws std::runtime_error If a successful query supplies no valid ordinal.
     */
    template<class Current, class Select>
    [[nodiscard]] NativeDeviceBindingResult bindNativeDevice(int requested, Current &&current, Select &&select)
    {
        if (requested < 0) throw std::invalid_argument("Native device binding requires an exact nonnegative ordinal");
        int observed = -1;
        const int query_status = static_cast<int>(std::invoke(std::forward<Current>(current), &observed));
        if (query_status != 0) return {NativeDeviceBindingStep::CurrentDeviceQuery, query_status, observed};
        if (observed < 0) throw std::runtime_error("Native current-device query succeeded without a valid ordinal");
        if (observed == requested) return {NativeDeviceBindingStep::AlreadySelected, 0, observed};
        const int select_status = static_cast<int>(std::invoke(std::forward<Select>(select), requested));
        return {NativeDeviceBindingStep::DeviceSelection, select_status, observed};
    }
}
