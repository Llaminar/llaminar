/**
 * @file GPUExpertTransfer.cpp
 * @brief Backend-neutral GPU expert weight transfer dispatch.
 */

#include "GPUExpertTransfer.h"
#include "GPUExpertTransferBackend.h"
#include "../../utils/Logger.h"

namespace llaminar2
{
    bool GPUExpertTransfer::transferExpert(
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
        if (!stream)
        {
            LOG_ERROR("[GPUExpertTransfer] Refusing GPU expert transfer on nullptr stream");
            return false;
        }

        if (src_device.is_cuda() && dst_device.is_cuda())
        {
#ifdef HAVE_CUDA
            return detail::transferExpertCUDABackend(
                src_ptrs, dst_ptrs, src_device, dst_device,
                vnni_bytes, scales_bytes, mins_bytes, emins_bytes, stream);
#else
            LOG_ERROR("[GPUExpertTransfer] CUDA support not available");
            return false;
#endif
        }

        if (src_device.is_rocm() && dst_device.is_rocm())
        {
#ifdef HAVE_ROCM
            return detail::transferExpertROCmBackend(
                src_ptrs, dst_ptrs, src_device, dst_device,
                vnni_bytes, scales_bytes, mins_bytes, emins_bytes, stream);
#else
            LOG_ERROR("[GPUExpertTransfer] ROCm support not available");
            return false;
#endif
        }

        LOG_ERROR("[GPUExpertTransfer] Same-backend GPU transfer required, got "
                  << src_device.to_string() << " -> " << dst_device.to_string());
        return false;
    }

    bool GPUExpertTransfer::transferExpert(
        const GpuExpertPackedDescriptor &src,
        const GpuExpertPackedDescriptor &dst,
        const DeviceId &src_device,
        const DeviceId &dst_device,
        void *stream)
    {
        if (!gpuExpertPackedDescriptorsCompatible(src, dst))
        {
            LOG_ERROR("[GPUExpertTransfer] Incompatible GPU expert packed descriptors"
                      << " src_valid=" << src.valid()
                      << " dst_valid=" << dst.valid()
                      << " src_n=" << src.n << " dst_n=" << dst.n
                      << " src_k=" << src.k << " dst_k=" << dst.k
                      << " src_codebook=" << static_cast<int>(src.codebook_id)
                      << " dst_codebook=" << static_cast<int>(dst.codebook_id));
            return false;
        }

        return transferExpert(src.ptrs, dst.ptrs, src_device, dst_device,
                              src.vnni_bytes, src.scales_bytes,
                              src.mins_bytes, src.emins_bytes, stream);
    }

    bool GPUExpertTransfer::activateStagedExpert(
        const GpuExpertPackedDescriptor &staged,
        const GpuExpertPackedDescriptor &active,
        const DeviceId &device,
        void *stream)
    {
        return transferExpert(staged, active, device, device, stream);
    }

    bool GPUExpertTransfer::activateStagedExperts(
        const std::vector<GpuExpertStagedActivation> &activations,
        const DeviceId &device,
        void *stream)
    {
        if (activations.empty())
            return true;
        if (!stream)
        {
            LOG_ERROR("[GPUExpertTransfer] Refusing staged expert activation batch on nullptr stream");
            return false;
        }

        for (const auto &activation : activations)
        {
            if (!activateStagedExpert(
                    activation.staged,
                    activation.active,
                    device,
                    stream))
            {
                return false;
            }
        }
        return true;
    }

}
