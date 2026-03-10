#include "CUDAWeightPacker.h"

#include "CUDAQuantisedGemmKernel.h"
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
        bool canPackNativePayloadCUDA(const TensorBase *tensor)
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

        bool packInt8ExpandedCUDA(const TensorBase *tensor, CUDAPackedWeights &out)
        {
            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());

            out.int8_data.assign(static_cast<size_t>(K) * N, int8_t{0});
            out.scales.assign(N, 1.0f);
            out.K = K;
            out.N = N;

            if (const auto *quant_accessor = dynamic_cast<const IINT8Unpackable *>(tensor);
                quant_accessor && (K % 32) == 0)
            {
#pragma omp parallel for schedule(static)
                for (int n = 0; n < N; ++n)
                {
                    out.scales[n] = quant_accessor->requantizeRowToInt8(
                        static_cast<size_t>(n),
                        static_cast<size_t>(K),
                        out.int8_data.data() + static_cast<size_t>(n) * K);
                }

                return true;
            }

            const float *h_weights_fp32 = tensor->data();
            if (!h_weights_fp32)
            {
                LOG_ERROR("[packInt8ExpandedCUDA] Failed to get FP32 data from tensor");
                return false;
            }

#pragma omp parallel for schedule(static)
            for (int n = 0; n < N; ++n)
            {
                float max_abs = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    max_abs = std::max(max_abs, std::abs(h_weights_fp32[n * K + k]));
                }

                const float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
                const float inv_scale = 1.0f / scale;
                out.scales[n] = scale;

                for (int k = 0; k < K; ++k)
                {
                    const float val = h_weights_fp32[n * K + k];
                    out.int8_data[n * K + k] = static_cast<int8_t>(
                        std::round(std::clamp(val * inv_scale, -127.0f, 127.0f)));
                }
            }

            return true;
        }

        bool packNativePayloadCUDA(const TensorBase *tensor, CUDAPackedWeights &out)
        {
            const auto *quant_accessor = dynamic_cast<const IINT8Unpackable *>(tensor);
            if (!quant_accessor)
            {
                LOG_ERROR("[packNativePayloadCUDA] Tensor type "
                          << tensorTypeName(tensor->native_type())
                          << " does not implement IINT8Unpackable");
                return false;
            }

            const auto *info = quant_accessor->vnniFormatInfo();
            if (!info)
            {
                LOG_ERROR("[packNativePayloadCUDA] Missing VNNI format info for tensor type "
                          << tensorTypeName(tensor->native_type()));
                return false;
            }

            const int N = static_cast<int>(tensor->rows());
            const int K = static_cast<int>(tensor->cols());
            if ((K % 32) != 0)
            {
                LOG_ERROR("[packNativePayloadCUDA] K=" << K << " not divisible by 32 for "
                                                       << tensorTypeName(tensor->native_type()));
                return false;
            }

            const int blocks_per_row = K / 32;
            out.native_payload.assign(static_cast<size_t>(blocks_per_row) * N * info->payload_bytes, uint8_t{0});
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
            ctx.payload_array = out.native_payload.data();
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
        for (auto &[device_id, upload] : device_uploads)
        {
            (void)device_id;
            if (upload.d_int8_data)
                cudaQuantGemm_freeDevice(upload.d_int8_data);
            if (upload.d_scales)
                cudaQuantGemm_freeDevice(upload.d_scales);
            if (upload.d_int8_data_tc_blocked)
                cudaQuantGemm_freeDevice(upload.d_int8_data_tc_blocked);
            if (upload.d_native_payload)
                cudaQuantGemm_freeDevice(upload.d_native_payload);
            if (upload.d_native_scales)
                cudaQuantGemm_freeDevice(upload.d_native_scales);
            if (upload.d_native_mins)
                cudaQuantGemm_freeDevice(upload.d_native_mins);
            if (upload.d_native_emins)
                cudaQuantGemm_freeDevice(upload.d_native_emins);
        }

        if (device_uploads.empty())
        {
            if (d_int8_data)
                cudaQuantGemm_freeDevice(d_int8_data);
            if (d_scales)
                cudaQuantGemm_freeDevice(d_scales);
            if (d_int8_data_tc_blocked)
                cudaQuantGemm_freeDevice(d_int8_data_tc_blocked);
            if (d_native_payload)
                cudaQuantGemm_freeDevice(d_native_payload);
            if (d_native_scales)
                cudaQuantGemm_freeDevice(d_native_scales);
            if (d_native_mins)
                cudaQuantGemm_freeDevice(d_native_mins);
            if (d_native_emins)
                cudaQuantGemm_freeDevice(d_native_emins);
        }
    }

    CUDAPackedWeightFamily classifyCUDAPackedWeightFamily(TensorType type)
    {
        if (isNativeVnniFormat(type))
        {
            return CUDAPackedWeightFamily::NativePayload;
        }

        return CUDAPackedWeightFamily::Int8Expanded;
    }

    const char *cudaPackedWeightFamilyName(CUDAPackedWeightFamily family)
    {
        switch (family)
        {
        case CUDAPackedWeightFamily::Int8Expanded:
            return "Int8Expanded";
        case CUDAPackedWeightFamily::NativePayload:
            return "NativePayload";
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

        out.preferred_family = classifyCUDAPackedWeightFamily(tensor->native_type());
        out.active_family = CUDAPackedWeightFamily::Int8Expanded;
        out.native_payload.clear();
        out.native_scales.clear();
        out.native_mins.clear();
        out.native_emins.clear();
        out.native_codebook_id = 0;
        out.native_blocks_per_row = 0;

        const int N = static_cast<int>(tensor->rows());
        const int K = static_cast<int>(tensor->cols());
        out.K = K;
        out.N = N;

        if (out.preferred_family == CUDAPackedWeightFamily::NativePayload && canPackNativePayloadCUDA(tensor))
        {
            if (packNativePayloadCUDA(tensor, out))
            {
                out.active_family = CUDAPackedWeightFamily::NativePayload;
            }
            else
            {
                LOG_WARN("[packWeightsToCUDA] NativePayload packing unavailable for "
                         << tensorTypeName(tensor->native_type()) << " " << N << "x" << K
                         << "; falling back to Int8Expanded");
            }
        }
        else if (out.preferred_family == CUDAPackedWeightFamily::NativePayload)
        {
            LOG_DEBUG("[packWeightsToCUDA] NativePayload not eligible for "
                      << tensorTypeName(tensor->native_type()) << " " << N << "x" << K
                      << "; using Int8Expanded as active family");
        }

        if (!packInt8ExpandedCUDA(tensor, out))
        {
            LOG_ERROR("[packWeightsToCUDA] Int8Expanded packing failed for "
                      << tensorTypeName(tensor->native_type()) << " " << N << "x" << K);
            return false;
        }

        LOG_DEBUG("[packWeightsToCUDA] Packed " << N << "x" << K
                             << " weights with active_family=" << cudaPackedWeightFamilyName(out.active_family)
                             << " preferred_family=" << cudaPackedWeightFamilyName(out.preferred_family)
                             << " source_type=" << tensorTypeName(tensor->native_type())
                             << " native_payload_bytes=" << out.native_payload.size()
                             << " int8_fallback_bytes=" << out.int8_data.size()
                             << ")");
        LOG_DEBUG("[packWeightsToCUDA] First 4 host scales: "
                  << out.scales[0] << "," << (N > 1 ? out.scales[1] : 0.f) << ","
                  << (N > 2 ? out.scales[2] : 0.f) << "," << (N > 3 ? out.scales[3] : 0.f));
        return true;
    }
} // namespace llaminar2::cuda