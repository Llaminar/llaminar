/**
 * @file GPUExpertTransferROCm.cpp
 * @brief ROCm backend for same-backend GPU expert packed transfer.
 */

#include "GPUExpertTransferBackend.h"
#include "../../utils/Logger.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

#include <map>
#include <mutex>
#include <utility>

namespace llaminar2::detail
{
    namespace
    {
        std::mutex peer_access_mutex;
        std::map<std::pair<int, int>, bool> peer_access_cache;
        std::mutex peer_enable_mutex;
        std::map<std::pair<int, int>, bool> peer_enable_cache;

        bool copyArrayPeerAsyncROCm(
            void *dst,
            int dst_ordinal,
            const void *src,
            int src_ordinal,
            size_t bytes,
            hipStream_t stream,
            bool peer_available)
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
                          << " bytes=" << bytes
                          << " peer_access=" << (peer_available ? "true" : "false")
                          << ")");
                return false;
            }
            return true;
        }
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
        const bool peer = canAccessPeerROCmBackend(src_device, dst_device);

        if (peer && src_ord != dst_ord)
            (void)enablePeerAccessROCmBackend(dst_device, src_device);

        auto hip_stream = static_cast<hipStream_t>(stream);
        bool success = true;
        if (!copyArrayPeerAsyncROCm(dst_ptrs.d_vnni, dst_ord,
                                    src_ptrs.d_vnni, src_ord,
                                    vnni_bytes, hip_stream, peer))
            success = false;
        if (success && !copyArrayPeerAsyncROCm(dst_ptrs.d_scales, dst_ord,
                                               src_ptrs.d_scales, src_ord,
                                               scales_bytes, hip_stream, peer))
            success = false;
        if (success && mins_bytes > 0 &&
            !copyArrayPeerAsyncROCm(dst_ptrs.d_mins, dst_ord,
                                    src_ptrs.d_mins, src_ord,
                                    mins_bytes, hip_stream, peer))
            success = false;
        if (success && emins_bytes > 0 &&
            !copyArrayPeerAsyncROCm(dst_ptrs.d_emins, dst_ord,
                                    src_ptrs.d_emins, src_ord,
                                    emins_bytes, hip_stream, peer))
            success = false;

        if (original_device >= 0)
            (void)hipSetDevice(original_device);

        if (success)
        {
            LOG_DEBUG("[GPUExpertTransfer] Transferred expert ROCm:" << src_ord
                                                                     << " -> ROCm:" << dst_ord
                                                                     << " vnni=" << vnni_bytes
                                                                     << "B scales=" << scales_bytes
                                                                     << "B mins=" << mins_bytes
                                                                     << "B emins=" << emins_bytes
                                                                     << "B"
                                                                     << (peer ? " (P2P)" : " (host-staged)"));
        }
        return success;
    }

    bool canAccessPeerROCmBackend(const DeviceId &src_device, const DeviceId &dst_device)
    {
        if (src_device == dst_device)
            return true;

        const auto key = std::make_pair(src_device.rocm_ordinal(), dst_device.rocm_ordinal());
        std::lock_guard<std::mutex> lock(peer_access_mutex);
        auto it = peer_access_cache.find(key);
        if (it != peer_access_cache.end())
            return it->second;

        int can_access = 0;
        hipError_t err = hipDeviceCanAccessPeer(
            &can_access,
            src_device.rocm_ordinal(),
            dst_device.rocm_ordinal());
        const bool result = err == hipSuccess && can_access != 0;
        peer_access_cache[key] = result;
        return result;
    }

    bool enablePeerAccessROCmBackend(
        const DeviceId &current_device,
        const DeviceId &peer_device)
    {
        if (current_device == peer_device)
            return true;

        const auto key = std::make_pair(current_device.rocm_ordinal(), peer_device.rocm_ordinal());
        std::lock_guard<std::mutex> lock(peer_enable_mutex);
        auto it = peer_enable_cache.find(key);
        if (it != peer_enable_cache.end())
            return it->second;

        if (!canAccessPeerROCmBackend(current_device, peer_device))
        {
            peer_enable_cache[key] = false;
            return false;
        }

        int original_device = -1;
        (void)hipGetDevice(&original_device);
        hipError_t set_err = hipSetDevice(current_device.rocm_ordinal());
        if (set_err != hipSuccess)
        {
            peer_enable_cache[key] = false;
            return false;
        }

        hipError_t err = hipDeviceEnablePeerAccess(peer_device.rocm_ordinal(), 0);
        if (original_device >= 0)
            (void)hipSetDevice(original_device);
        const bool result = err == hipSuccess || err == hipErrorPeerAccessAlreadyEnabled;
        peer_enable_cache[key] = result;
        return result;
    }
}
#endif
