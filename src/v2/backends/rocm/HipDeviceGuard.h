/**
 * @file HipDeviceGuard.h
 * @brief Checked HIP thread-device binding without an ordinal shadow cache.
 *
 * Raw runtime and vendor calls can change the current device independently.
 * Ordinary binding reads actual cheap thread-local native state; execution
 * boundaries may explicitly force selection. No reset/invalidation discipline
 * or process-lifetime mutable device cache is required.
 */
#pragma once
#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif
namespace llaminar2
{
    /** @brief Bind the actual HIP thread state while preserving native failure status. */
    class HipDeviceGuard
    {
    public:
        /**
         * @brief Query actual thread state and select only a differing device.
         * @param device_idx Exact nonnegative target ordinal.
         * @return Original HIP query/selection status; zero means success.
         * @throws std::invalid_argument For an invalid ordinal.
         * @throws std::runtime_error For an incomplete successful native observation.
         * @throws std::logic_error When this binary was built without ROCm.
         */
        static int setDevice(int device_idx);
        /**
         * @brief Select unconditionally at an explicit execution boundary.
         * @param device_idx Exact native target ordinal.
         * @return Original HIP selection status, without updating a second authority.
         * @throws std::logic_error When this binary was built without ROCm.
         */
        static int forceSetDevice(int device_idx);
    };
}
