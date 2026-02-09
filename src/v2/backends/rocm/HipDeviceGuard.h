/**
 * @file HipDeviceGuard.h
 * @brief Thread-local HIP device tracking to eliminate redundant hipSetDevice calls.
 *
 * During decode, every ROCm kernel wrapper calls hipSetDevice(device_idx) even though
 * the device doesn't change between stages. Each hipSetDevice costs ~1-5µs on MI50,
 * and with 339+ calls per token, this adds 0.3-1.7ms of pure overhead.
 *
 * This utility tracks the current device per-thread and skips redundant calls.
 */
#pragma once

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{

    /// Thread-local HIP device state tracker.
    /// Call setDevice() instead of hipSetDevice() in kernel wrappers.
    /// Call resetTracking() when entering a new execution context.
    class HipDeviceGuard
    {
    public:
        /// Set the HIP device, skipping the call if already on the correct device.
        /// Returns hipSuccess on success or skip, error code on failure.
        static int setDevice(int device_idx)
        {
#ifdef HAVE_ROCM
            if (device_idx == current_device_)
                return 0; // Already on correct device — skip hipSetDevice
            hipError_t err = hipSetDevice(device_idx);
            if (err == hipSuccess)
                current_device_ = device_idx;
            return static_cast<int>(err);
#else
            (void)device_idx;
            return 0;
#endif
        }

        /// Reset tracking (call at the start of a new execution context).
        /// Forces the next setDevice() call to actually call hipSetDevice.
        static void resetTracking()
        {
            current_device_ = -1;
        }

        /// Set device and update tracking without the skip optimization.
        /// Use this at execution boundaries (e.g., start of forward pass).
        static int forceSetDevice(int device_idx)
        {
#ifdef HAVE_ROCM
            hipError_t err = hipSetDevice(device_idx);
            if (err == hipSuccess)
                current_device_ = device_idx;
            return static_cast<int>(err);
#else
            (void)device_idx;
            return 0;
#endif
        }

    private:
        static thread_local int current_device_;
    };

} // namespace llaminar2
