/**
 * @file ROCmRMSNormKernelT.cpp
 * @brief ROCm RMSNorm kernel implementation
 *
 * Host-side adapter that calls HIP kernels via extern "C" wrappers.
 */

#include "ROCmRMSNormKernelT.h"
#include "../../../backends/DeviceId.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include "utils/ROCmKernelProfiler.h"
#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>

// Extern "C" declarations for HIP kernels
extern "C" bool hipOps_rmsnorm_fp32(
    const float *input, const float *gamma, float *output,
    int rows, int cols, float epsilon, int device_idx, void *stream);

extern "C" bool hipOps_rmsnorm_bf16(
    const uint16_t *input, const float *gamma, uint16_t *output,
    int rows, int cols, float epsilon, int device_idx, void *stream);

extern "C" bool hipOps_rmsnorm_fp16(
    const uint16_t *input, const float *gamma, uint16_t *output,
    int rows, int cols, float epsilon, int device_idx, void *stream);

namespace
{
    /**
     * @brief Record one successful production ROCm RMSNorm verifier launch.
     *
     * The HIP kernels use one workgroup per logical row. Grouping M=2..4 rows
     * consequently keeps every row's M=1 reduction order intact while issuing
     * a single kernel launch. This counter is omitted for serial M=1 witnesses
     * so the integration gate can reject hidden row replay or multiple launches.
     *
     * @param tensor_format Native activation/output representation.
     * @param verifier_rows Number of MTP verifier rows in this launch.
     * @param cols Number of values reduced and normalized per row.
     * @param device ROCm ordinal on which the grouped kernel was launched.
     */
    void recordROCmGroupedRMSNormCall(
        const char *tensor_format,
        int verifier_rows,
        int cols,
        int device)
    {
        if (verifier_rows < 2 || verifier_rows > 4)
            return;

        llaminar2::PerfStatsCollector::addCounter(
            "kernel",
            "rocm_rmsnorm_grouped_verifier_rows_calls",
            1.0,
            "verifier",
            llaminar2::DeviceId::rocm(device).to_string(),
            {{"tensor_format", tensor_format},
             {"verifier_rows", std::to_string(verifier_rows)},
             {"cols", std::to_string(cols)},
             {"capture_mode", llaminar2::isGraphCaptureActive() ? "graph_capture" : "direct"},
             {"row_mapping", "one_block_per_row"},
             {"invocation_policy", "single_grouped_launch"}});
    }
} // namespace

namespace llaminar2
{
    namespace rocm
    {

        // ============================================================================
        // FP32 Implementation
        // ============================================================================

        bool ROCmRMSNormKernelT<ActivationPrecision::FP32>::apply(
            const float *input,
            const float *weight,
            float *output,
            int rows,
            int cols,
            float epsilon,
            bool /*use_bf16*/,
            const IMPIContext * /*mpi_ctx*/,
            int device_idx)
        {
            // For FP32, assume pointers are already device pointers when device_idx >= 0
            if (device_idx < 0)
            {
                LOG_ERROR("ROCmRMSNormKernelT<FP32>::apply requires device_idx >= 0");
                return false;
            }
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::FP32>::apply_typed(
            const float *d_input,
            const float *d_gamma,
            float *d_output,
            int rows,
            int cols,
            float epsilon,
            int device_idx)
        {
            return hipOps_rmsnorm_fp32(d_input, d_gamma, d_output, rows, cols, epsilon, device_idx, gpu_stream_);
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::FP32>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP32>] Null tensor pointer");
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP32>] apply_tensor requires an explicit non-null HIP stream");
                return false;
            }
            if (input->native_type() != TensorType::FP32 || output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP32>] Type mismatch: expected FP32 input/output");
                return false;
            }

            // Cast to FP32Tensor for GPU operations
            auto *input_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(output);

            if (!input_fp32 || !weight_fp32 || !output_fp32)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP32>] Dynamic cast to FP32Tensor failed");
                return false;
            }

            // Coherence handled by DeviceGraphExecutor - tensors are already on GPU
            // Get device pointers
            const float *d_input_ptr = static_cast<const float *>(input_fp32->gpu_data_ptr());
            const float *d_weight_ptr = static_cast<const float *>(weight_fp32->gpu_data_ptr());
            float *d_output_ptr = static_cast<float *>(output_fp32->gpu_data_ptr());

            ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                ROCmKernelType::RMS_NORM,
                static_cast<hipStream_t>(gpu_stream_));
            const bool ok = hipOps_rmsnorm_fp32(
                d_input_ptr, d_weight_ptr, d_output_ptr,
                rows, cols, epsilon, device_idx, gpu_stream_);
            if (ok)
                recordROCmGroupedRMSNormCall("FP32", rows, cols, device_idx);
            // No sync is required; executor stream ordering owns completion.
            return ok;
        }

        // ============================================================================
        // BF16 Implementation
        // ============================================================================

        bool ROCmRMSNormKernelT<ActivationPrecision::BF16>::apply(
            const float * /*input*/,
            const float * /*weight*/,
            float * /*output*/,
            int /*rows*/,
            int /*cols*/,
            float /*epsilon*/,
            bool /*use_bf16*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("ROCmRMSNormKernelT<BF16>::apply(float*) not supported. Use apply_bf16().");
            return false;
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::BF16>::apply_bf16(
            const uint16_t *input,
            const float *weight,
            uint16_t *output,
            int rows,
            int cols,
            float epsilon,
            int device_idx)
        {
            if (device_idx < 0)
            {
                LOG_ERROR("ROCmRMSNormKernelT<BF16>::apply_bf16 requires device_idx >= 0");
                return false;
            }
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::BF16>::apply_typed(
            const uint16_t *d_input,
            const float *d_gamma,
            uint16_t *d_output,
            int rows,
            int cols,
            float epsilon,
            int device_idx)
        {
            return hipOps_rmsnorm_bf16(d_input, d_gamma, d_output, rows, cols, epsilon, device_idx, gpu_stream_);
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::BF16>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<BF16>] Null tensor pointer");
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<BF16>] apply_tensor requires an explicit non-null HIP stream");
                return false;
            }
            if (input->native_type() != TensorType::BF16 || output->native_type() != TensorType::BF16)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<BF16>] Type mismatch: expected BF16 input/output");
                return false;
            }

            // Cast to BF16Tensor for GPU operations
            auto *input_bf16 = const_cast<BF16Tensor *>(dynamic_cast<const BF16Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *output_bf16 = dynamic_cast<BF16Tensor *>(output);

            if (!input_bf16 || !weight_fp32 || !output_bf16)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<BF16>] Dynamic cast failed");
                return false;
            }

            // Get device pointers
            const uint16_t *d_input_ptr = static_cast<const uint16_t *>(input_bf16->gpu_data_ptr());
            const float *d_weight_ptr = static_cast<const float *>(weight_fp32->gpu_data_ptr());
            uint16_t *d_output_ptr = static_cast<uint16_t *>(output_bf16->gpu_data_ptr());

            ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                ROCmKernelType::RMS_NORM,
                static_cast<hipStream_t>(gpu_stream_));
            const bool ok = hipOps_rmsnorm_bf16(
                d_input_ptr, d_weight_ptr, d_output_ptr,
                rows, cols, epsilon, device_idx, gpu_stream_);
            if (ok)
                recordROCmGroupedRMSNormCall("BF16", rows, cols, device_idx);
            // No sync is required; executor stream ordering owns completion.
            return ok;
        }

        // ============================================================================
        // FP16 Implementation
        // ============================================================================

        bool ROCmRMSNormKernelT<ActivationPrecision::FP16>::apply(
            const float * /*input*/,
            const float * /*weight*/,
            float * /*output*/,
            int /*rows*/,
            int /*cols*/,
            float /*epsilon*/,
            bool /*use_bf16*/,
            const IMPIContext * /*mpi_ctx*/,
            int /*device_idx*/)
        {
            LOG_ERROR("ROCmRMSNormKernelT<FP16>::apply(float*) not supported. Use apply_fp16().");
            return false;
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::FP16>::apply_fp16(
            const uint16_t *input,
            const float *weight,
            uint16_t *output,
            int rows,
            int cols,
            float epsilon,
            int device_idx)
        {
            if (device_idx < 0)
            {
                LOG_ERROR("ROCmRMSNormKernelT<FP16>::apply_fp16 requires device_idx >= 0");
                return false;
            }
            return apply_typed(input, weight, output, rows, cols, epsilon, device_idx);
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::FP16>::apply_typed(
            const uint16_t *d_input,
            const float *d_gamma,
            uint16_t *d_output,
            int rows,
            int cols,
            float epsilon,
            int device_idx)
        {
            return hipOps_rmsnorm_fp16(d_input, d_gamma, d_output, rows, cols, epsilon, device_idx, gpu_stream_);
        }

        bool ROCmRMSNormKernelT<ActivationPrecision::FP16>::apply_tensor(
            const TensorBase *input,
            const TensorBase *weight,
            TensorBase *output,
            int rows, int cols,
            float epsilon,
            const IMPIContext *mpi_ctx,
            int device_idx)
        {
            (void)mpi_ctx;
            if (!input || !weight || !output)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP16>] Null tensor pointer");
                return false;
            }
            if (!gpu_stream_)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP16>] apply_tensor requires an explicit non-null HIP stream");
                return false;
            }
            if (input->native_type() != TensorType::FP16 || output->native_type() != TensorType::FP16)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP16>] Type mismatch: expected FP16 input/output");
                return false;
            }

            // Cast to FP16Tensor for GPU operations
            auto *input_fp16 = const_cast<FP16Tensor *>(dynamic_cast<const FP16Tensor *>(input));
            auto *weight_fp32 = const_cast<FP32Tensor *>(dynamic_cast<const FP32Tensor *>(weight));
            auto *output_fp16 = dynamic_cast<FP16Tensor *>(output);

            if (!input_fp16 || !weight_fp32 || !output_fp16)
            {
                LOG_ERROR("[ROCmRMSNormKernelT<FP16>] Dynamic cast failed");
                return false;
            }

            // Get device pointers
            const uint16_t *d_input_ptr = static_cast<const uint16_t *>(input_fp16->gpu_data_ptr());
            const float *d_weight_ptr = static_cast<const float *>(weight_fp32->gpu_data_ptr());
            uint16_t *d_output_ptr = static_cast<uint16_t *>(output_fp16->gpu_data_ptr());

            ROCM_KERNEL_PROFILE_SCOPE_STREAM(
                ROCmKernelType::RMS_NORM,
                static_cast<hipStream_t>(gpu_stream_));
            const bool ok = hipOps_rmsnorm_fp16(
                d_input_ptr, d_weight_ptr, d_output_ptr,
                rows, cols, epsilon, device_idx, gpu_stream_);
            if (ok)
                recordROCmGroupedRMSNormCall("FP16", rows, cols, device_idx);
            // No sync is required; executor stream ordering owns completion.
            return ok;
        }

    } // namespace rocm
} // namespace llaminar2
