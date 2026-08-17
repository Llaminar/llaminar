/**
 * @file GPUExpertTransferROCm.cpp
 * @brief ROCm backend for same-backend GPU expert packed transfer.
 */

#include "GPUExpertTransferBackend.h"
#include "../../backends/rocm/HipDeviceGuard.h"
#include "../../utils/Logger.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

namespace llaminar2::detail
{
    namespace
    {
        bool copyArrayDeviceAsyncROCm(
            void *dst,
            int dst_ordinal,
            const void *src,
            int src_ordinal,
            size_t bytes,
            hipStream_t stream)
        {
            if (bytes == 0 || dst == nullptr || src == nullptr)
                return true;

            hipError_t err = hipMemcpyPeerAsync(
                dst, dst_ordinal, src, src_ordinal, bytes, stream);
            if (err != hipSuccess)
            {
                LOG_ERROR("[GPUExpertTransfer] hipMemcpyPeerAsync failed: "
                          << hipGetErrorString(err)
                          << " (src=" << src_ordinal
                          << " dst=" << dst_ordinal
                          << " bytes=" << bytes << ")");
                return false;
            }
            return true;
        }
    }

    int currentROCmDeviceOrdinal() noexcept
    {
        int ordinal = -1;
        return hipGetDevice(&ordinal) == hipSuccess ? ordinal : -1;
    }

    bool restoreROCmDeviceOrdinal(int ordinal) noexcept
    {
        return ordinal >= 0 &&
               HipDeviceGuard::forceSetDevice(ordinal) ==
                   static_cast<int>(hipSuccess);
    }

    bool transferExpertROCmBackend(
        const GPUExpertPointers &src_ptrs,
        const GPUExpertPointers &dst_ptrs,
        const DeviceId &src_device,
        const DeviceId &dst_device,
        size_t vnni_bytes,
        size_t scales_bytes,
        size_t mins_bytes,
        size_t emins_bytes,
        void *stream)
    {
        int original_device = -1;
        (void)hipGetDevice(&original_device);

        const int src_ord = src_device.rocm_ordinal();
        const int dst_ord = dst_device.rocm_ordinal();

        /* Peer submissions execute on a destination-owned auxiliary stream;
         * never inherit an unrelated current device from the caller thread. */
        hipError_t select_err = hipSetDevice(dst_ord);
        if (select_err != hipSuccess)
        {
            LOG_ERROR("[GPUExpertTransfer] hipSetDevice failed for destination "
                      << dst_ord << ": " << hipGetErrorString(select_err));
            return false;
        }

        auto hip_stream = static_cast<hipStream_t>(stream);
        bool success = true;
        if (!copyArrayDeviceAsyncROCm(dst_ptrs.d_vnni, dst_ord,
                                    src_ptrs.d_vnni, src_ord,
                                    vnni_bytes, hip_stream))
            success = false;
        if (success && !copyArrayDeviceAsyncROCm(dst_ptrs.d_scales, dst_ord,
                                               src_ptrs.d_scales, src_ord,
                                               scales_bytes, hip_stream))
            success = false;
        if (success && mins_bytes > 0 &&
            !copyArrayDeviceAsyncROCm(dst_ptrs.d_mins, dst_ord,
                                    src_ptrs.d_mins, src_ord,
                                    mins_bytes, hip_stream))
            success = false;
        if (success && emins_bytes > 0 &&
            !copyArrayDeviceAsyncROCm(dst_ptrs.d_emins, dst_ord,
                                    src_ptrs.d_emins, src_ord,
                                    emins_bytes, hip_stream))
            success = false;

        if (original_device >= 0)
            (void)restoreROCmDeviceOrdinal(original_device);

        if (success)
        {
            LOG_DEBUG("[GPUExpertTransfer] Transferred expert ROCm:" << src_ord
                                                                     << " -> ROCm:" << dst_ord
                                                                     << " vnni=" << vnni_bytes
                                                                     << "B scales=" << scales_bytes
                                                                     << "B mins=" << mins_bytes
                                                                     << "B emins=" << emins_bytes
                                                                     << "B");
        }
        return success;
    }
}
#endif
