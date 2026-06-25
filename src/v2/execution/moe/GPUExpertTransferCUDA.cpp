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
        bool copyArrayPeerAsyncCUDA(
            void *dst,
            int dst_ordinal,
            const void *src,
            int src_ordinal,
            size_t bytes,
            cudaStream_t stream,
            bool peer_available)
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
                          << " bytes=" << bytes
                          << " peer_access=" << (peer_available ? "true" : "false")
                          << ")");
                return false;
            }
            return true;
        }
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
        const bool peer = canAccessPeerCUDABackend(src_device, dst_device);

        if (peer && src_ord != dst_ord)
            (void)enablePeerAccessCUDABackend(dst_device, src_device);

        auto cuda_stream = static_cast<cudaStream_t>(stream);
        bool success = true;
        if (!copyArrayPeerAsyncCUDA(dst_ptrs.d_vnni, dst_ord,
                                    src_ptrs.d_vnni, src_ord,
                                    vnni_bytes, cuda_stream, peer))
            success = false;
        if (success && !copyArrayPeerAsyncCUDA(dst_ptrs.d_scales, dst_ord,
                                               src_ptrs.d_scales, src_ord,
                                               scales_bytes, cuda_stream, peer))
            success = false;
        if (success && mins_bytes > 0 &&
            !copyArrayPeerAsyncCUDA(dst_ptrs.d_mins, dst_ord,
                                    src_ptrs.d_mins, src_ord,
                                    mins_bytes, cuda_stream, peer))
            success = false;
        if (success && emins_bytes > 0 &&
            !copyArrayPeerAsyncCUDA(dst_ptrs.d_emins, dst_ord,
                                    src_ptrs.d_emins, src_ord,
                                    emins_bytes, cuda_stream, peer))
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
                                                                     << "B"
                                                                     << (peer ? " (P2P)" : " (host-staged)"));
        }
        return success;
    }

    bool canAccessPeerCUDABackend(const DeviceId &src_device, const DeviceId &dst_device)
    {
        if (src_device == dst_device)
            return true;
        int can_access = 0;
        cudaError_t err = cudaDeviceCanAccessPeer(
            &can_access,
            src_device.cuda_ordinal(),
            dst_device.cuda_ordinal());
        return err == cudaSuccess && can_access != 0;
    }

    bool enablePeerAccessCUDABackend(
        const DeviceId &current_device,
        const DeviceId &peer_device)
    {
        if (current_device == peer_device)
            return true;

        int original_device = -1;
        (void)cudaGetDevice(&original_device);
        cudaError_t set_err = cudaSetDevice(current_device.cuda_ordinal());
        if (set_err != cudaSuccess)
            return false;

        cudaError_t err = cudaDeviceEnablePeerAccess(peer_device.cuda_ordinal(), 0);
        if (original_device >= 0)
            (void)cudaSetDevice(original_device);
        return err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled;
    }
}
#endif
