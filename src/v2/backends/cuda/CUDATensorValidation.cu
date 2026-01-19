/**
 * @file CUDATensorValidation.cu
 * @brief CUDA GPU-accelerated tensor validation kernels
 *
 * CUDA kernels for NaN/Inf/Zero detection without D2H transfer.
 *
 * @author David Sanftenberg
 */

#include "../../tensors/TensorValidation.h"
#include "../../utils/Logger.h"
#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <algorithm>

namespace llaminar2
{
    // =========================================================================
    // Device-side Validation Result (mirrors TensorValidationResult)
    // =========================================================================

    struct DeviceValidationResult
    {
        unsigned int has_nan;
        unsigned int has_inf;
        unsigned int appears_zero;
        unsigned int nan_count;
        unsigned int inf_count;
        unsigned int zero_count;
        unsigned int total_checked;
        float sample_min;
        float sample_max;
    };

    // =========================================================================
    // CUDA Validation Kernels
    // =========================================================================

    /**
     * @brief FP32 validation kernel - parallel reduction for NaN/Inf detection
     */
    __global__ void validateFP32KernelCUDA(
        const float *__restrict__ data,
        size_t num_elements,
        DeviceValidationResult *__restrict__ result)
    {
        __shared__ unsigned int s_nan_count;
        __shared__ unsigned int s_inf_count;
        __shared__ unsigned int s_zero_count;
        __shared__ unsigned int s_has_nonzero;
        __shared__ float s_min;
        __shared__ float s_max;

        if (threadIdx.x == 0)
        {
            s_nan_count = 0;
            s_inf_count = 0;
            s_zero_count = 0;
            s_has_nonzero = 0;
            s_min = FLT_MAX;
            s_max = -FLT_MAX;
        }
        __syncthreads();

        unsigned int local_nan = 0;
        unsigned int local_inf = 0;
        unsigned int local_zero = 0;
        float local_min = FLT_MAX;
        float local_max = -FLT_MAX;

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < num_elements; i += stride)
        {
            float val = data[i];

            if (isnan(val))
            {
                local_nan++;
            }
            else if (isinf(val))
            {
                local_inf++;
            }
            else
            {
                if (val == 0.0f)
                {
                    local_zero++;
                }
                local_min = fminf(local_min, val);
                local_max = fmaxf(local_max, val);
            }
        }

        if (local_nan > 0)
            atomicAdd(&s_nan_count, local_nan);
        if (local_inf > 0)
            atomicAdd(&s_inf_count, local_inf);
        if (local_zero > 0)
            atomicAdd(&s_zero_count, local_zero);
        if (local_min < FLT_MAX || local_max > -FLT_MAX)
        {
            atomicAdd(&s_has_nonzero, 1u);
        }

        // Atomic min/max for floats using integer reinterpretation
        if (local_min < FLT_MAX)
        {
            atomicMin(reinterpret_cast<int *>(&s_min), __float_as_int(local_min));
        }
        if (local_max > -FLT_MAX)
        {
            atomicMax(reinterpret_cast<int *>(&s_max), __float_as_int(local_max));
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            atomicAdd(&result->nan_count, s_nan_count);
            atomicAdd(&result->inf_count, s_inf_count);
            atomicAdd(&result->zero_count, s_zero_count);

            if (s_nan_count > 0)
                atomicOr(&result->has_nan, 1u);
            if (s_inf_count > 0)
                atomicOr(&result->has_inf, 1u);
            if (s_has_nonzero > 0)
                atomicAnd(&result->appears_zero, 0u);

            atomicMin(reinterpret_cast<int *>(&result->sample_min), __float_as_int(s_min));
            atomicMax(reinterpret_cast<int *>(&result->sample_max), __float_as_int(s_max));
        }
    }

    /**
     * @brief BF16 validation kernel
     */
    __global__ void validateBF16KernelCUDA(
        const uint16_t *__restrict__ data,
        size_t num_elements,
        DeviceValidationResult *__restrict__ result)
    {
        __shared__ unsigned int s_nan_count;
        __shared__ unsigned int s_inf_count;
        __shared__ unsigned int s_zero_count;
        __shared__ unsigned int s_has_nonzero;

        if (threadIdx.x == 0)
        {
            s_nan_count = 0;
            s_inf_count = 0;
            s_zero_count = 0;
            s_has_nonzero = 0;
        }
        __syncthreads();

        unsigned int local_nan = 0;
        unsigned int local_inf = 0;
        unsigned int local_zero = 0;

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < num_elements; i += stride)
        {
            uint16_t bf16 = data[i];
            uint16_t exp = (bf16 >> 7) & 0xFF;
            uint16_t mant = bf16 & 0x7F;

            if (exp == 0xFF)
            {
                if (mant != 0)
                    local_nan++;
                else
                    local_inf++;
            }
            else if (bf16 == 0 || bf16 == 0x8000)
            {
                local_zero++;
            }
        }

        if (local_nan > 0)
            atomicAdd(&s_nan_count, local_nan);
        if (local_inf > 0)
            atomicAdd(&s_inf_count, local_inf);
        if (local_zero > 0)
            atomicAdd(&s_zero_count, local_zero);
        if (local_nan == 0 && local_inf == 0 && local_zero < (num_elements / (blockDim.x * gridDim.x) + 1))
        {
            atomicAdd(&s_has_nonzero, 1u);
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            atomicAdd(&result->nan_count, s_nan_count);
            atomicAdd(&result->inf_count, s_inf_count);
            atomicAdd(&result->zero_count, s_zero_count);

            if (s_nan_count > 0)
                atomicOr(&result->has_nan, 1u);
            if (s_inf_count > 0)
                atomicOr(&result->has_inf, 1u);
            if (s_has_nonzero > 0)
                atomicAnd(&result->appears_zero, 0u);
        }
    }

    /**
     * @brief FP16 validation kernel
     */
    __global__ void validateFP16KernelCUDA(
        const uint16_t *__restrict__ data,
        size_t num_elements,
        DeviceValidationResult *__restrict__ result)
    {
        __shared__ unsigned int s_nan_count;
        __shared__ unsigned int s_inf_count;
        __shared__ unsigned int s_zero_count;
        __shared__ unsigned int s_has_nonzero;

        if (threadIdx.x == 0)
        {
            s_nan_count = 0;
            s_inf_count = 0;
            s_zero_count = 0;
            s_has_nonzero = 0;
        }
        __syncthreads();

        unsigned int local_nan = 0;
        unsigned int local_inf = 0;
        unsigned int local_zero = 0;

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < num_elements; i += stride)
        {
            uint16_t fp16 = data[i];
            uint16_t exp = (fp16 >> 10) & 0x1F;
            uint16_t mant = fp16 & 0x3FF;

            if (exp == 0x1F)
            {
                if (mant != 0)
                    local_nan++;
                else
                    local_inf++;
            }
            else if (fp16 == 0 || fp16 == 0x8000)
            {
                local_zero++;
            }
        }

        if (local_nan > 0)
            atomicAdd(&s_nan_count, local_nan);
        if (local_inf > 0)
            atomicAdd(&s_inf_count, local_inf);
        if (local_zero > 0)
            atomicAdd(&s_zero_count, local_zero);
        if (local_nan == 0 && local_inf == 0 && local_zero < (num_elements / (blockDim.x * gridDim.x) + 1))
        {
            atomicAdd(&s_has_nonzero, 1u);
        }

        __syncthreads();

        if (threadIdx.x == 0)
        {
            atomicAdd(&result->nan_count, s_nan_count);
            atomicAdd(&result->inf_count, s_inf_count);
            atomicAdd(&result->zero_count, s_zero_count);

            if (s_nan_count > 0)
                atomicOr(&result->has_nan, 1u);
            if (s_inf_count > 0)
                atomicOr(&result->has_inf, 1u);
            if (s_has_nonzero > 0)
                atomicAnd(&result->appears_zero, 0u);
        }
    }

    // =========================================================================
    // CUDA Tensor Validator Implementation
    // =========================================================================

    class CUDATensorValidator : public ITensorValidator
    {
    public:
        CUDATensorValidator()
        {
            cudaError_t err = cudaMalloc(&d_result_, sizeof(DeviceValidationResult));
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDATensorValidator] Failed to allocate device result buffer");
                d_result_ = nullptr;
            }
        }

        ~CUDATensorValidator() override
        {
            if (d_result_)
            {
                cudaFree(d_result_);
                d_result_ = nullptr;
            }
        }

        bool validateFP32Async(const void *device_ptr,
                               size_t num_elements,
                               int device_id) override
        {
            if (!d_result_ || !device_ptr || num_elements == 0)
                return false;

            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess)
                return false;

            DeviceValidationResult init = {};
            init.appears_zero = 1;
            init.sample_min = FLT_MAX;
            init.sample_max = -FLT_MAX;

            err = cudaMemcpy(d_result_, &init, sizeof(DeviceValidationResult), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                return false;

            const int block_size = 256;
            const int max_blocks = 1024;
            int num_blocks = std::min(max_blocks, (int)((num_elements + block_size - 1) / block_size));

            validateFP32KernelCUDA<<<num_blocks, block_size>>>(
                static_cast<const float *>(device_ptr),
                num_elements,
                d_result_);

            last_num_elements_ = num_elements;
            return true;
        }

        bool validateBF16Async(const void *device_ptr,
                               size_t num_elements,
                               int device_id) override
        {
            if (!d_result_ || !device_ptr || num_elements == 0)
                return false;

            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess)
                return false;

            DeviceValidationResult init = {};
            init.appears_zero = 1;

            err = cudaMemcpy(d_result_, &init, sizeof(DeviceValidationResult), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                return false;

            const int block_size = 256;
            const int max_blocks = 1024;
            int num_blocks = std::min(max_blocks, (int)((num_elements + block_size - 1) / block_size));

            validateBF16KernelCUDA<<<num_blocks, block_size>>>(
                static_cast<const uint16_t *>(device_ptr),
                num_elements,
                d_result_);

            last_num_elements_ = num_elements;
            return true;
        }

        bool validateFP16Async(const void *device_ptr,
                               size_t num_elements,
                               int device_id) override
        {
            if (!d_result_ || !device_ptr || num_elements == 0)
                return false;

            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess)
                return false;

            DeviceValidationResult init = {};
            init.appears_zero = 1;

            err = cudaMemcpy(d_result_, &init, sizeof(DeviceValidationResult), cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
                return false;

            const int block_size = 256;
            const int max_blocks = 1024;
            int num_blocks = std::min(max_blocks, (int)((num_elements + block_size - 1) / block_size));

            validateFP16KernelCUDA<<<num_blocks, block_size>>>(
                static_cast<const uint16_t *>(device_ptr),
                num_elements,
                d_result_);

            last_num_elements_ = num_elements;
            return true;
        }

        bool getResult(TensorValidationResult &result) override
        {
            if (!d_result_)
                return false;

            cudaError_t err = cudaDeviceSynchronize();
            if (err != cudaSuccess)
                return false;

            DeviceValidationResult d_res;
            err = cudaMemcpy(&d_res, d_result_, sizeof(DeviceValidationResult), cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
                return false;

            result.has_nan = (d_res.has_nan != 0);
            result.has_inf = (d_res.has_inf != 0);
            result.appears_zero = (d_res.appears_zero != 0);
            result.valid = !result.has_nan && !result.has_inf;
            result.nan_count = d_res.nan_count;
            result.inf_count = d_res.inf_count;
            result.zero_count = d_res.zero_count;
            result.total_checked = static_cast<uint32_t>(last_num_elements_);
            result.sample_min = d_res.sample_min;
            result.sample_max = d_res.sample_max;

            return true;
        }

    private:
        DeviceValidationResult *d_result_ = nullptr;
        size_t last_num_elements_ = 0;
    };

    // =========================================================================
    // Factory Function
    // =========================================================================

    static CUDATensorValidator *g_cuda_validator = nullptr;

    ITensorValidator *getCUDATensorValidator()
    {
        if (!g_cuda_validator)
        {
            g_cuda_validator = new CUDATensorValidator();
        }
        return g_cuda_validator;
    }

} // namespace llaminar2

// C linkage export for cross-TU factory
extern "C" llaminar2::ITensorValidator *llaminar2_getCUDATensorValidator()
{
    return llaminar2::getCUDATensorValidator();
}
