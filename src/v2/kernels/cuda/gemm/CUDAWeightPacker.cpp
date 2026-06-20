#include "CUDAWeightPacker.h"

#include "CUDAQuantisedGemmKernel.h"
#include "CUDADeviceWorkspace.h"
#include "tensors/TensorClasses.h"
#include "tensors/TensorType.h"
#include "tensors/VnniPackContext.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>

namespace llaminar2::cuda
{
    extern "C"
    {
        void cudaQuantGemm_freeDevice(void *d_ptr);
    }

    namespace
    {
        bool canPackNativeVNNICUDA(const TensorBase *tensor)
        {
            if (!tensor)
            {
                return false;
            }

            const auto *quant_accessor = dynamic_cast<const IINT8Unpackable *>(tensor);
            if (!quant_accessor)
            {
                return false;
            }

            const auto *info = quant_accessor->vnniFormatInfo();
            if (!info)
            {
                return false;
            }

            return (tensor->cols() % 32) == 0;
        }

        bool packNativeVNNICUDA(const TensorBase *tensor, CUDAPackedWeights &out)
        {
            const auto *quant_accessor = dynamic_cast<const IINT8Unpackable *>(tensor);
            if (!quant_accessor)
            {
                LOG_ERROR("[packNativeVNNICUDA] Tensor type "
                          << tensorTypeName(tensor->native_type())
                          << " does not implement IINT8Unpackable");
                return false;
            }

            const auto *info = quant_accessor->vnniFormatInfo();
            if (!info)
            {
                LOG_ERROR("[packNativeVNNICUDA] Missing VNNI format info for tensor type "
                          << tensorTypeName(tensor->native_type()));
                return false;
            }

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());
            if ((K % 32) != 0)
            {
                LOG_ERROR("[packNativeVNNICUDA] K=" << K << " not divisible by 32 for "
                                                    << tensorTypeName(tensor->native_type()));
                return false;
            }

            const int blocks_per_row = K / 32;
            out.native_vnni.assign(static_cast<size_t>(blocks_per_row) * N * info->payload_bytes, uint8_t{0});
            out.native_scales.assign(static_cast<size_t>(blocks_per_row) * N, uint16_t{0});
            out.native_mins.clear();
            out.native_emins.clear();
            if (info->is_asymmetric)
            {
                out.native_mins.assign(static_cast<size_t>(blocks_per_row) * N, uint16_t{0});
            }
            if (info->has_emins)
            {
                out.native_emins.assign(static_cast<size_t>(blocks_per_row) * N, uint32_t{0});
            }
            out.native_codebook_id = info->codebook_id;
            out.native_blocks_per_row = static_cast<uint32_t>(blocks_per_row);

            VnniPackContext ctx{};
            ctx.raw_bytes = nullptr;
            ctx.N = N;
            ctx.K = K;
            ctx.blocks_per_row = blocks_per_row;
            ctx.payload_bytes = info->payload_bytes;
            ctx.payload_array = out.native_vnni.data();
            ctx.scales_array = out.native_scales.data();
            ctx.mins_array = info->is_asymmetric ? out.native_mins.data() : nullptr;
            ctx.emins_array = info->has_emins ? out.native_emins.data() : nullptr;

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    quant_accessor->packVnniBlock(ctx, n, b);
                }
            }

            return true;
        }
    }

    CUDAPackedWeights::~CUDAPackedWeights()
    {
        // Free row-major transpose (per-weight, used by ROWPAR GEMV)
        if (rowmajor_)
        {
            cudaRowMajorWeights_destroy(rowmajor_);
            rowmajor_ = nullptr;
        }

        for (auto &[device_id, upload] : device_uploads)
        {
            (void)device_id;
            if (upload.d_native_vnni)
                cudaQuantGemm_freeDevice(upload.d_native_vnni);
            if (upload.d_native_scales)
                cudaQuantGemm_freeDevice(upload.d_native_scales);
            if (upload.d_native_mins)
                cudaQuantGemm_freeDevice(upload.d_native_mins);
            if (upload.d_native_emins)
                cudaQuantGemm_freeDevice(upload.d_native_emins);
        }

        if (device_uploads.empty())
        {
            if (d_native_vnni)
                cudaQuantGemm_freeDevice(d_native_vnni);
            if (d_native_scales)
                cudaQuantGemm_freeDevice(d_native_scales);
            if (d_native_mins)
                cudaQuantGemm_freeDevice(d_native_mins);
            if (d_native_emins)
                cudaQuantGemm_freeDevice(d_native_emins);
        }
    }

    CUDAPackedWeightFamily classifyCUDAPackedWeightFamily(TensorType /*type*/)
    {
        return CUDAPackedWeightFamily::NativeVNNI;
    }

    const char *cudaPackedWeightFamilyName(CUDAPackedWeightFamily family)
    {
        switch (family)
        {
        case CUDAPackedWeightFamily::NativeVNNI:
            return "NativeVNNI";
        default:
            return "Unknown";
        }
    }

    bool packWeightsToCUDA(const TensorBase *tensor, CUDAPackedWeights &out)
    {
        if (!tensor)
        {
            LOG_ERROR("[packWeightsToCUDA] Null tensor");
            return false;
        }

        out.preferred_family = CUDAPackedWeightFamily::NativeVNNI;
        out.active_family = CUDAPackedWeightFamily::NativeVNNI;
        out.native_vnni.clear();
        out.native_scales.clear();
        out.native_mins.clear();
        out.native_emins.clear();
        out.native_codebook_id = 0;
        out.native_blocks_per_row = 0;

        const int N = static_cast<int>(tensor->rows());
        const int K = static_cast<int>(tensor->cols());
        out.K = K;
        out.N = N;

        if (!canPackNativeVNNICUDA(tensor))
        {
            LOG_ERROR("[packWeightsToCUDA] NativeVNNI packing not available for "
                      << tensorTypeName(tensor->native_type()) << " " << N << "x" << K
                      << " (no Int8Expanded fallback — TC/CUTLASS paths have been removed)");
            return false;
        }

        if (!packNativeVNNICUDA(tensor, out))
        {
            LOG_ERROR("[packWeightsToCUDA] NativeVNNI packing failed for "
                      << tensorTypeName(tensor->native_type()) << " " << N << "x" << K);
            return false;
        }

        LOG_DEBUG("[packWeightsToCUDA] Packed " << N << "x" << K
                                                << " weights with active_family=NativeVNNI"
                                                << " source_type=" << tensorTypeName(tensor->native_type())
                                                << " native_vnni_bytes=" << out.native_vnni.size()
                                                << ")");
        return true;
    }

    // =========================================================================
    // MoE Batch Packing — pack all experts into one contiguous slab
    // =========================================================================

    extern "C"
    {
        bool cudaQuantGemm_uploadRawBytes(const void *h_src, void **d_dst, size_t bytes, int cuda_device_id);
        void cudaQuantGemm_freeDevice(void *d_ptr);
    }

    MoEBatchPackedWeightsCUDA::~MoEBatchPackedWeightsCUDA()
    {
        for (auto &[device_id, upload] : device_uploads)
        {
            (void)device_id;
            if (upload.d_vnni)
                cudaQuantGemm_freeDevice(upload.d_vnni);
            if (upload.d_scales)
                cudaQuantGemm_freeDevice(upload.d_scales);
            if (upload.d_mins)
                cudaQuantGemm_freeDevice(upload.d_mins);
            if (upload.d_emins)
                cudaQuantGemm_freeDevice(upload.d_emins);
        }
    }

    bool MoEBatchPackedWeightsCUDA::uploadToDevice(int cuda_device_id)
    {
        std::lock_guard<std::mutex> lock(upload_mutex);

        auto it = device_uploads.find(cuda_device_id);
        if (it != device_uploads.end())
            return true; // Already uploaded

        DeviceUpload upload;

        auto uploadArray = [&](const auto &host_vec, auto **d_ptr) -> bool
        {
            *d_ptr = nullptr;
            if (host_vec.empty())
                return true;
            using T = typename std::remove_reference_t<decltype(host_vec)>::value_type;
            const size_t bytes = host_vec.size() * sizeof(T);
            void *d_mem = nullptr;
            if (!cudaQuantGemm_uploadRawBytes(host_vec.data(), &d_mem, bytes, cuda_device_id))
                return false;
            *d_ptr = reinterpret_cast<decltype(*d_ptr)>(d_mem);
            return true;
        };

        if (!uploadArray(all_vnni, &upload.d_vnni) ||
            !uploadArray(all_scales, &upload.d_scales) ||
            !uploadArray(all_mins, &upload.d_mins) ||
            !uploadArray(all_emins, &upload.d_emins))
        {
            if (upload.d_vnni) cudaQuantGemm_freeDevice(upload.d_vnni);
            if (upload.d_scales) cudaQuantGemm_freeDevice(upload.d_scales);
            if (upload.d_mins) cudaQuantGemm_freeDevice(upload.d_mins);
            if (upload.d_emins) cudaQuantGemm_freeDevice(upload.d_emins);
            return false;
        }

        device_uploads.emplace(cuda_device_id, upload);
        return true;
    }

    MoEBatchPackedWeightsCUDA::DeviceUpload
    MoEBatchPackedWeightsCUDA::getExpertDevicePointers(int cuda_device_id, int expert_id) const
    {
        auto it = device_uploads.find(cuda_device_id);
        if (it == device_uploads.end())
            return {};

        const auto &base = it->second;
        DeviceUpload expert;
        expert.d_vnni = base.d_vnni ? base.d_vnni + expert_id * vnni_bytes_per_expert : nullptr;
        expert.d_scales = base.d_scales ? base.d_scales + expert_id * scales_per_expert : nullptr;
        expert.d_mins = base.d_mins ? base.d_mins + expert_id * mins_per_expert : nullptr;
        expert.d_emins = base.d_emins ? base.d_emins + expert_id * emins_per_expert : nullptr;
        return expert;
    }

    void MoEBatchPackedWeightsCUDA::freeHostBuffers()
    {
        const size_t freed = all_vnni.capacity() +
                             all_scales.capacity() * sizeof(uint16_t) +
                             all_mins.capacity() * sizeof(uint16_t) +
                             all_emins.capacity() * sizeof(uint32_t);
        all_vnni.clear();
        all_vnni.shrink_to_fit();
        all_scales.clear();
        all_scales.shrink_to_fit();
        all_mins.clear();
        all_mins.shrink_to_fit();
        all_emins.clear();
        all_emins.shrink_to_fit();
        if (freed > 0)
        {
            LOG_DEBUG("[MoEBatchPackedWeightsCUDA] Released host buffers: "
                      << (freed / (1024 * 1024)) << " MB");
        }
    }

    std::shared_ptr<MoEBatchPackedWeightsCUDA> packMoEExpertsCUDA(
        const std::vector<std::shared_ptr<TensorBase>> &expert_views,
        int num_experts, int rows_per_expert)
    {
        if (expert_views.empty() || num_experts <= 0 || rows_per_expert <= 0)
        {
            LOG_ERROR("[packMoEExpertsCUDA] Invalid arguments");
            return nullptr;
        }

        // Get format info from first expert view
        const auto *quant = dynamic_cast<const IINT8Unpackable *>(expert_views[0].get());
        if (!quant)
        {
            LOG_ERROR("[packMoEExpertsCUDA] Expert view does not implement IINT8Unpackable");
            return nullptr;
        }
        const auto *info = quant->vnniFormatInfo();
        if (!info)
        {
            LOG_ERROR("[packMoEExpertsCUDA] Expert view has no VNNI format info");
            return nullptr;
        }

        const int K = static_cast<int>(expert_views[0]->cols());
        if ((K % 32) != 0)
        {
            LOG_ERROR("[packMoEExpertsCUDA] K=" << K << " not divisible by 32");
            return nullptr;
        }

        const int blocks_per_row = K / 32;

        auto batch = std::make_shared<MoEBatchPackedWeightsCUDA>();
        batch->num_experts = num_experts;
        batch->rows_per_expert = rows_per_expert;
        batch->K = K;
        batch->blocks_per_row = blocks_per_row;
        batch->codebook_id = info->codebook_id;

        // Per-expert sizes
        batch->vnni_bytes_per_expert = static_cast<size_t>(blocks_per_row) * rows_per_expert * info->payload_bytes;
        batch->scales_per_expert = static_cast<size_t>(blocks_per_row) * rows_per_expert;
        batch->mins_per_expert = info->is_asymmetric ? batch->scales_per_expert : 0;
        batch->emins_per_expert = info->has_emins ? batch->scales_per_expert : 0;

        // Allocate contiguous slabs
        batch->all_vnni.assign(static_cast<size_t>(num_experts) * batch->vnni_bytes_per_expert, uint8_t{0});
        batch->all_scales.assign(static_cast<size_t>(num_experts) * batch->scales_per_expert, uint16_t{0});
        if (info->is_asymmetric)
            batch->all_mins.assign(static_cast<size_t>(num_experts) * batch->mins_per_expert, uint16_t{0});
        if (info->has_emins)
            batch->all_emins.assign(static_cast<size_t>(num_experts) * batch->emins_per_expert, uint32_t{0});

        // Pack each expert into its region of the slab
        for (int e = 0; e < num_experts; ++e)
        {
            const auto *eq = dynamic_cast<const IINT8Unpackable *>(expert_views[e].get());
            if (!eq)
            {
                LOG_ERROR("[packMoEExpertsCUDA] Expert " << e << " not IINT8Unpackable");
                return nullptr;
            }

            VnniPackContext ctx{};
            ctx.raw_bytes = nullptr;
            ctx.N = rows_per_expert;
            ctx.K = K;
            ctx.blocks_per_row = blocks_per_row;
            ctx.payload_bytes = info->payload_bytes;
            ctx.payload_array = batch->all_vnni.data() + e * batch->vnni_bytes_per_expert;
            ctx.scales_array = batch->all_scales.data() + e * batch->scales_per_expert;
            ctx.mins_array = info->is_asymmetric
                                 ? batch->all_mins.data() + e * batch->mins_per_expert
                                 : nullptr;
            ctx.emins_array = info->has_emins
                                  ? batch->all_emins.data() + e * batch->emins_per_expert
                                  : nullptr;

#pragma omp parallel for schedule(static)
            for (int n = 0; n < rows_per_expert; ++n)
            {
                for (int b = 0; b < blocks_per_row; ++b)
                {
                    eq->packVnniBlock(ctx, n, b);
                }
            }
        }

        LOG_DEBUG("[packMoEExpertsCUDA] Packed " << num_experts << " experts ("
                 << rows_per_expert << "x" << K << " each), total slab "
                 << (batch->all_vnni.size() / (1024 * 1024)) << " MB vnni + "
                 << (batch->all_scales.size() * 2 / (1024 * 1024)) << " MB scales");

        return batch;
    }

} // namespace llaminar2::cuda