/**
 * @file GPUExpertTransferROCm.cpp
 * @brief ROCm expert migration and prepared capture-safe physical byte transport.
 *
 * Migration retains its original explicit peer API and ambient-device lifecycle.
 * Prepared copies use UVA asynchronous D2D on the authenticated destination
 * worker: the operation must become a native graph node, never eager setup work.
 * Neither path retries or selects an alternate transport after a native failure.
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

    /** @brief Prepared UVA device copy records a native memcpy node during capture.
     * The explicit PeerAsync API is deliberately retained only by legacy
     * migration below: CUDA rejects it in capture, while HIP may enqueue it
     * eagerly without a graph node. This is a distinct prepared path, never a
     * capture-state retry or an eager substitute.
     */
    bool copyPreparedDevicePeerBytesROCmBackend(void* destination,int destination_ordinal,const void* source,int source_ordinal,std::size_t bytes,void* destination_stream)
    {
        if(!destination || !source || !destination_stream)return false;
        if(!bytes)return true;
        // TransferEngine has authenticated the exact original destination
        // worker/stream; capture must not mutate that native context binding.
        auto status=hipMemcpyAsync(destination,source,bytes,hipMemcpyDeviceToDevice,static_cast<hipStream_t>(destination_stream));
        if(status!=hipSuccess){
            LOG_ERROR("[PreparedDevicePeerCopy] hipMemcpyAsync(D2D) failed: " << hipGetErrorString(status)
                << " src=" << source_ordinal << " dst=" << destination_ordinal << " bytes=" << bytes);
            return false;
        }
        return true;
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

    bool prepareDirectPeerAccessROCmBackend(
        const DeviceId &source,
        const DeviceId &destination) noexcept
    {
        if (!source.is_rocm() || !destination.is_rocm() ||
            source == destination)
        {
            return source == destination && source.is_rocm();
        }

        // Setup owns only its own AlreadyEnabled status. Never overwrite or
        // consume an unrelated error left by an earlier owner on this thread.
        const hipError_t previous_error = hipPeekAtLastError();
        if (previous_error != hipSuccess)
        {
            LOG_ERROR("[GPUExpertTransfer] Direct-peer setup found an unconsumed ROCm error: "
                      << hipGetErrorString(previous_error)
                      << " status=" << static_cast<int>(previous_error));
            return false;
        }

        int original_device = -1;
        if (hipGetDevice(&original_device) != hipSuccess)
            return false;

        const int source_ordinal = source.rocm_ordinal();
        const int destination_ordinal = destination.rocm_ordinal();
        bool prepared = false;
        if (HipDeviceGuard::forceSetDevice(destination_ordinal) ==
            static_cast<int>(hipSuccess))
        {
            int can_access = 0;
            const hipError_t query = hipDeviceCanAccessPeer(
                &can_access,
                destination_ordinal,
                source_ordinal);
            if (query == hipSuccess && can_access != 0)
            {
                const hipError_t enable =
                    hipDeviceEnablePeerAccess(source_ordinal, 0);
                prepared = enable == hipSuccess;
                if (enable == hipErrorPeerAccessAlreadyEnabled)
                {
                    // Successful native APIs do not retire this expected
                    // status. Prove it is precisely ours before consuming it;
                    // a different error remains visible and rejects setup.
                    const auto owned_error = hipPeekAtLastError();
                    if (owned_error == hipErrorPeerAccessAlreadyEnabled)
                    {
                        const auto retired = hipGetLastError();
                        prepared = retired == hipErrorPeerAccessAlreadyEnabled;
                        if (!prepared)
                            LOG_ERROR("[GPUExpertTransfer] Expected peer-access status changed during retirement: "
                                      << hipGetErrorString(retired));
                    }
                    else
                    {
                        LOG_ERROR("[GPUExpertTransfer] Peer-access setup cannot consume unrelated ROCm error: "
                                  << hipGetErrorString(owned_error));
                    }
                }
            }
        }

        if (!restoreROCmDeviceOrdinal(original_device))
        {
            LOG_ERROR("[GPUExpertTransfer] Could not restore ROCm device "
                      << original_device
                      << " after direct-peer setup");
            return false;
        }
        return prepared;
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
