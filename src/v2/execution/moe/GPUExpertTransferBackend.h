#pragma once

#include "GPUExpertTransfer.h"

namespace llaminar2::detail
{
#ifdef HAVE_CUDA
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

    bool canAccessPeerCUDABackend(const DeviceId &src_device, const DeviceId &dst_device);
    bool enablePeerAccessCUDABackend(const DeviceId &current_device, const DeviceId &peer_device);
#endif

#ifdef HAVE_ROCM
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

    bool canAccessPeerROCmBackend(const DeviceId &src_device, const DeviceId &dst_device);
    bool enablePeerAccessROCmBackend(const DeviceId &current_device, const DeviceId &peer_device);
#endif
}
