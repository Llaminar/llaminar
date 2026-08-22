/**
 * @file GPUExpertTransferCUDA.cpp
 * @brief CUDA backend for same-backend GPU expert packed transfer.
 */

#include "GPUExpertTransferBackend.h"
#include "../../utils/Logger.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

namespace llaminar2::detail
{
    namespace
    {
        bool copyArrayDeviceAsyncCUDA(
            void *dst,
            int dst_ordinal,
            const void *src,
            int src_ordinal,
            size_t bytes,
            cudaStream_t stream)
        {
            if (bytes == 0 || dst == nullptr || src == nullptr)
                return true;

            if (src_ordinal == dst_ordinal)
            {
                cudaError_t err = cudaMemcpyAsync(
                    dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
                if (err != cudaSuccess)
                {
                    LOG_ERROR("[GPUExpertTransfer] cudaMemcpy D2D failed: "
                              << cudaGetErrorString(err)
                              << " (device=" << src_ordinal << " bytes=" << bytes << ")");
                    return false;
                }
                return true;
            }

            cudaError_t err = cudaMemcpyPeerAsync(
                dst, dst_ordinal, src, src_ordinal, bytes, stream);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[GPUExpertTransfer] cudaMemcpyPeerAsync failed: "
                          << cudaGetErrorString(err)
                          << " (src=" << src_ordinal
                          << " dst=" << dst_ordinal
                          << " bytes=" << bytes << ")");
                return false;
            }
            return true;
        }
    }

    int currentCUDADeviceOrdinal() noexcept
    {
        int ordinal = -1;
        return cudaGetDevice(&ordinal) == cudaSuccess ? ordinal : -1;
    }

    bool restoreCUDADeviceOrdinal(int ordinal) noexcept
    {
        return ordinal >= 0 && cudaSetDevice(ordinal) == cudaSuccess;
    }

    bool prepareDirectPeerAccessCUDABackend(
        const DeviceId &source,
        const DeviceId &destination) noexcept
    {
        if (!source.is_cuda() || !destination.is_cuda() ||
            source == destination)
        {
            return source == destination && source.is_cuda();
        }

        int original_device = -1;
        if (cudaGetDevice(&original_device) != cudaSuccess)
            return false;

        const int source_ordinal = source.cuda_ordinal();
        const int destination_ordinal = destination.cuda_ordinal();
        bool prepared = false;
        if (cudaSetDevice(destination_ordinal) == cudaSuccess)
        {
            int can_access = 0;
            const cudaError_t query = cudaDeviceCanAccessPeer(
                &can_access,
                destination_ordinal,
                source_ordinal);
            if (query == cudaSuccess && can_access != 0)
            {
                const cudaError_t enable =
                    cudaDeviceEnablePeerAccess(source_ordinal, 0);
                prepared = enable == cudaSuccess ||
                           enable == cudaErrorPeerAccessAlreadyEnabled;
            }
        }

        if (cudaSetDevice(original_device) != cudaSuccess)
        {
            LOG_ERROR("[GPUExpertTransfer] Could not restore CUDA device "
                      << original_device
                      << " after direct-peer setup");
            return false;
        }
        return prepared;
    }

    bool transferExpertCUDABackend(
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
        (void)cudaGetDevice(&original_device);

        const int src_ord = src_device.cuda_ordinal();
        const int dst_ord = dst_device.cuda_ordinal();

        /* The supplied auxiliary stream is destination-owned. Select that
         * runtime context explicitly so peer submission never depends on the
         * maintenance thread's incidental current device. */
        cudaError_t select_err = cudaSetDevice(dst_ord);
        if (select_err != cudaSuccess)
        {
            LOG_ERROR("[GPUExpertTransfer] cudaSetDevice failed for destination "
                      << dst_ord << ": " << cudaGetErrorString(select_err));
            return false;
        }

        auto cuda_stream = static_cast<cudaStream_t>(stream);
        bool success = true;
        if (!copyArrayDeviceAsyncCUDA(dst_ptrs.d_vnni, dst_ord,
                                    src_ptrs.d_vnni, src_ord,
                                    vnni_bytes, cuda_stream))
            success = false;
        if (success && !copyArrayDeviceAsyncCUDA(dst_ptrs.d_scales, dst_ord,
                                               src_ptrs.d_scales, src_ord,
                                               scales_bytes, cuda_stream))
            success = false;
        if (success && mins_bytes > 0 &&
            !copyArrayDeviceAsyncCUDA(dst_ptrs.d_mins, dst_ord,
                                    src_ptrs.d_mins, src_ord,
                                    mins_bytes, cuda_stream))
            success = false;
        if (success && emins_bytes > 0 &&
            !copyArrayDeviceAsyncCUDA(dst_ptrs.d_emins, dst_ord,
                                    src_ptrs.d_emins, src_ord,
                                    emins_bytes, cuda_stream))
            success = false;

        if (original_device >= 0)
            (void)cudaSetDevice(original_device);

        if (success)
        {
            LOG_DEBUG("[GPUExpertTransfer] Transferred expert CUDA:" << src_ord
                                                                     << " -> CUDA:" << dst_ord
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
