/**
 * @file GPUExpertTransferBackend.h
 * @brief Private accelerator-runtime boundary for packed expert transfers.
 *
 * The public transfer API is backend-neutral and may be compiled with CUDA and
 * ROCm enabled together.  Runtime headers cannot safely coexist in one C++
 * translation unit, so this boundary owns runtime-specific submission and
 * ambient-device state access behind ordinary C++ declarations.
 */

#pragma once

#include "GPUExpertTransfer.h"

namespace llaminar2::detail
{
#ifdef HAVE_CUDA
    /** @return Calling thread's current CUDA ordinal, or -1 on failure. */
    int currentCUDADeviceOrdinal() noexcept;

    /**
     * @brief Restore a calling thread's CUDA runtime device.
     * @return True when the exact ordinal became current.
     */
    bool restoreCUDADeviceOrdinal(int ordinal) noexcept;

    /** Submit one separated NativeVNNI expert copy on a CUDA stream. */
    bool transferExpertCUDABackend(
        const GPUExpertPointers &src_ptrs,
        const GPUExpertPointers &dst_ptrs,
        const DeviceId &src_device,
        const DeviceId &dst_device,
        size_t vnni_bytes,
        size_t scales_bytes,
        size_t mins_bytes,
        size_t emins_bytes,
        void *stream);

#endif

#ifdef HAVE_ROCM
    /** @return Calling thread's current ROCm ordinal, or -1 on failure. */
    int currentROCmDeviceOrdinal() noexcept;

    /**
     * @brief Restore ROCm runtime and HipDeviceGuard state together.
     * @return True when the exact ordinal became current.
     */
    bool restoreROCmDeviceOrdinal(int ordinal) noexcept;

    /** Submit one separated NativeVNNI expert copy on a ROCm stream. */
    bool transferExpertROCmBackend(
        const GPUExpertPointers &src_ptrs,
        const GPUExpertPointers &dst_ptrs,
        const DeviceId &src_device,
        const DeviceId &dst_device,
        size_t vnni_bytes,
        size_t scales_bytes,
        size_t mins_bytes,
        size_t emins_bytes,
        void *stream);

#endif
}
