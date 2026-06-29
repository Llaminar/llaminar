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

#endif
}
