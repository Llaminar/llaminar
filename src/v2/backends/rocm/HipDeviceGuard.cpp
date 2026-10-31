/**
 * @file HipDeviceGuard.cpp
 * @brief Actual native current-device binding for HIP infrastructure callers.
 *
 * The shared checked selector authenticates current state on every call and
 * invokes hipSetDevice only on a real mismatch. External native selections,
 * alternating collective kinds and context retirement require no cache repair.
 */
#include "HipDeviceGuard.h"
#include "backends/NativeDeviceBinding.h"
#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{
    int HipDeviceGuard::setDevice(int device_idx)
    {
#ifdef HAVE_ROCM
        return bindNativeDevice(device_idx,
            [](int *observed) { return hipGetDevice(observed); },
            [](int selected) { return hipSetDevice(selected); }).native_status;
#else
        (void)device_idx;
        throw std::logic_error("HIP device binding is unavailable in a build without ROCm");
#endif
    }
    int HipDeviceGuard::forceSetDevice(int device_idx)
    {
#ifdef HAVE_ROCM
        return static_cast<int>(hipSetDevice(device_idx));
#else
        (void)device_idx;
        throw std::logic_error("HIP device binding is unavailable in a build without ROCm");
#endif
    }
}
