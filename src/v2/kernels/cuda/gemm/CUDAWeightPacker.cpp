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
} // namespace llaminar2::cuda