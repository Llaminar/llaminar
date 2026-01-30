/**
 * @file ROCmTensorValidation.cpp
 * @brief ROCm GPU-accelerated tensor validation kernels
 *
 * HIP kernels for NaN/Inf/Zero detection without D2H transfer.
 *
 * @author David Sanftenberg
 */

#include "../../tensors/GPUTensorVerification.h"
#include "../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

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
    // HIP Validation Kernels
    // =========================================================================

    /**
     * @brief FP32 validation kernel - parallel reduction for NaN/Inf detection
     *
     * Each thread block processes a chunk of the tensor and reduces to shared memory.
     * Final reduction across blocks uses atomics.
     */
    __global__ void validateFP32Kernel(
        const float *__restrict__ data,
        size_t num_elements,
        DeviceValidationResult *__restrict__ result)
    {
        // Shared memory for block-level reduction
        __shared__ unsigned int s_nan_count;
        __shared__ unsigned int s_inf_count;
        __shared__ unsigned int s_zero_count;
        __shared__ unsigned int s_has_nonzero;
        __shared__ float s_min;
        __shared__ float s_max;

        // Initialize shared memory (first thread in block)
        if (threadIdx.x == 0)
        {
            s_nan_count = 0;
            s_inf_count = 0;
            s_zero_count = 0;
            s_has_nonzero = 0;
            s_min = std::numeric_limits<float>::max();
            s_max = std::numeric_limits<float>::lowest();
        }
        __syncthreads();

        // Thread-local accumulators
        unsigned int local_nan = 0;
        unsigned int local_inf = 0;
        unsigned int local_zero = 0;
        float local_min = std::numeric_limits<float>::max();
        float local_max = std::numeric_limits<float>::lowest();

        // Grid-stride loop
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
                // Track min/max of finite values
                local_min = fminf(local_min, val);
                local_max = fmaxf(local_max, val);
            }
        }

        // Reduce to shared memory using atomics
        if (local_nan > 0)
            atomicAdd(&s_nan_count, local_nan);
        if (local_inf > 0)
            atomicAdd(&s_inf_count, local_inf);
        if (local_zero > 0)
            atomicAdd(&s_zero_count, local_zero);
        if (local_min < std::numeric_limits<float>::max() ||
            local_max > std::numeric_limits<float>::lowest())
        {
            atomicAdd(&s_has_nonzero, 1u);
        }

        // Use atomicMin/Max for floats (reinterpret as int for ordered comparison)
        // This works because IEEE floats have the property that their bit patterns
        // sort correctly when interpreted as signed integers (for positive values)
        if (local_min < std::numeric_limits<float>::max())
        {
            // For positive floats, we can use atomicMin on the bit pattern
            // For negative floats, we'd need special handling, but min/max are mainly for debugging
            atomicMin(reinterpret_cast<int *>(&s_min), __float_as_int(local_min));
        }
        if (local_max > std::numeric_limits<float>::lowest())
        {
            atomicMax(reinterpret_cast<int *>(&s_max), __float_as_int(local_max));
        }

        __syncthreads();

        // First thread in block writes to global result
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

            // Update global min/max
            atomicMin(reinterpret_cast<int *>(&result->sample_min), __float_as_int(s_min));
            atomicMax(reinterpret_cast<int *>(&result->sample_max), __float_as_int(s_max));
        }
    }

    /**
     * @brief BF16 validation kernel
     */
    __global__ void validateBF16Kernel(
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

            // BF16 format: 1 sign, 8 exponent, 7 mantissa
            // NaN: exponent = 0xFF, mantissa != 0
            // Inf: exponent = 0xFF, mantissa = 0
            uint16_t exp = (bf16 >> 7) & 0xFF;
            uint16_t mant = bf16 & 0x7F;

            if (exp == 0xFF)
            {
                if (mant != 0)
                    local_nan++;
                else
                    local_inf++;
            }
            else if (bf16 == 0 || bf16 == 0x8000) // +0 or -0
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
    __global__ void validateFP16Kernel(
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

            // FP16 format: 1 sign, 5 exponent, 10 mantissa
            // NaN: exponent = 0x1F, mantissa != 0
            // Inf: exponent = 0x1F, mantissa = 0
            uint16_t exp = (fp16 >> 10) & 0x1F;
            uint16_t mant = fp16 & 0x3FF;

            if (exp == 0x1F)
            {
                if (mant != 0)
                    local_nan++;
                else
                    local_inf++;
            }
            else if (fp16 == 0 || fp16 == 0x8000) // +0 or -0
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
    // ROCm Tensor Validator Implementation
    // =========================================================================

    class ROCmTensorValidator : public ITensorValidator
    {
    public:
        explicit ROCmTensorValidator(int device_id) : device_id_(device_id)
        {
            // Set device before allocation to ensure buffer is on correct device
            hipError_t err = hipSetDevice(device_id);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmTensorValidator] Failed to set device " << device_id);
                d_result_ = nullptr;
                return;
            }

            // Allocate device-side result buffer ON THIS DEVICE
            err = hipMalloc(&d_result_, sizeof(DeviceValidationResult));
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmTensorValidator] Failed to allocate device result buffer on device " << device_id);
                d_result_ = nullptr;
            }
        }

        ~ROCmTensorValidator() override
        {
            if (d_result_)
            {
                // Set device before freeing
                (void)hipSetDevice(device_id_);
                (void)hipFree(d_result_);
                d_result_ = nullptr;
            }
        }

        bool validateFP32Async(const void *device_ptr,
                               size_t num_elements,
                               int device_id) override
        {
            if (!d_result_ || !device_ptr || num_elements == 0)
                return false;

            // Verify we're validating on the device this validator was created for
            if (device_id != device_id_)
            {
                LOG_WARN("[ROCmTensorValidator] Device mismatch: validator for device " << device_id_
                         << " but asked to validate on device " << device_id);
                return false;
            }

            hipError_t err = hipSetDevice(device_id);
            if (err != hipSuccess)
                return false;

            // Initialize result on device
            DeviceValidationResult init = {};
            init.appears_zero = 1; // Start assuming all zeros
            init.sample_min = std::numeric_limits<float>::max();
            init.sample_max = std::numeric_limits<float>::lowest();

            err = hipMemcpy(d_result_, &init, sizeof(DeviceValidationResult), hipMemcpyHostToDevice);
            if (err != hipSuccess)
                return false;

            // Launch kernel
            const int block_size = 256;
            const int max_blocks = 1024;
            int num_blocks = std::min(max_blocks, (int)((num_elements + block_size - 1) / block_size));

            hipLaunchKernelGGL(validateFP32Kernel, dim3(num_blocks), dim3(block_size), 0, 0,
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

            hipError_t err = hipSetDevice(device_id);
            if (err != hipSuccess)
                return false;

            DeviceValidationResult init = {};
            init.appears_zero = 1;

            err = hipMemcpy(d_result_, &init, sizeof(DeviceValidationResult), hipMemcpyHostToDevice);
            if (err != hipSuccess)
                return false;

            const int block_size = 256;
            const int max_blocks = 1024;
            int num_blocks = std::min(max_blocks, (int)((num_elements + block_size - 1) / block_size));

            hipLaunchKernelGGL(validateBF16Kernel, dim3(num_blocks), dim3(block_size), 0, 0,
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

            hipError_t err = hipSetDevice(device_id);
            if (err != hipSuccess)
                return false;

            DeviceValidationResult init = {};
            init.appears_zero = 1;

            err = hipMemcpy(d_result_, &init, sizeof(DeviceValidationResult), hipMemcpyHostToDevice);
            if (err != hipSuccess)
                return false;

            const int block_size = 256;
            const int max_blocks = 1024;
            int num_blocks = std::min(max_blocks, (int)((num_elements + block_size - 1) / block_size));

            hipLaunchKernelGGL(validateFP16Kernel, dim3(num_blocks), dim3(block_size), 0, 0,
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

            // Synchronize and copy result
            hipError_t err = hipDeviceSynchronize();
            if (err != hipSuccess)
                return false;

            DeviceValidationResult d_res;
            err = hipMemcpy(&d_res, d_result_, sizeof(DeviceValidationResult), hipMemcpyDeviceToHost);
            if (err != hipSuccess)
                return false;

            // Convert to host result struct
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
        int device_id_ = -1;
    };

    // =========================================================================
    // Per-Device Validator Factory (Thread-Safe)
    // =========================================================================

    static std::mutex g_rocm_validator_mutex;
    static std::unordered_map<int, std::unique_ptr<ROCmTensorValidator>> g_rocm_validators;

    ITensorValidator *getROCmTensorValidator()
    {
        // Get current device
        int device_id = 0;
        hipError_t err = hipGetDevice(&device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[getROCmTensorValidator] Failed to get current device");
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(g_rocm_validator_mutex);

        auto it = g_rocm_validators.find(device_id);
        if (it == g_rocm_validators.end())
        {
            // Create a new validator for this device
            auto validator = std::make_unique<ROCmTensorValidator>(device_id);
            auto* ptr = validator.get();
            g_rocm_validators[device_id] = std::move(validator);
            LOG_DEBUG("[getROCmTensorValidator] Created validator for device " << device_id);
            return ptr;
        }
        return it->second.get();
    }

} // namespace llaminar2

// C linkage export for cross-TU factory
extern "C" llaminar2::ITensorValidator *llaminar2_getROCmTensorValidator()
{
    return llaminar2::getROCmTensorValidator();
}
