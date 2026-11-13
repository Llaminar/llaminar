/**
 * @file ROCmTensorValidation.cpp
 * @brief Stream-ordered ROCm tensor validation kernels.
 *
 * HIP kernels for NaN/Inf/zero detection without tensor materialization on the
 * host. Validation uses persistent per-device diagnostic storage and the exact
 * producer stream; only the compact result crosses D2H after an event.
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
#include <stdexcept>
#include <string>
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
        unsigned int invalid_extent;
        float sample_min;
        float sample_max;
    };

    static_assert(sizeof(DeviceValidationResult) <= 64,
                  "Validation publishes one compact diagnostic record");

    /**
     * @brief Initialize the existing transaction and acquire its original frontier.
     * @param result Persistent per-device diagnostic result, exclusively borrowed.
     * @param extent Checked static geometry and stream-ordered native count words.
     *
     * Invalid native metadata suppresses payload access and is published as an
     * error, distinct from valid-empty. This replaces the old initializer rather
     * than adding a count kernel or a separate host observation boundary.
     */
    __global__ void initializeValidationResultROCm(DeviceValidationResult *result,
                                                TensorValidationExtent extent)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            *result = {};
            result->appears_zero = 1;
            result->sample_min = std::numeric_limits<float>::max();
            result->sample_max = std::numeric_limits<float>::lowest();
            if(extent.usesProjectionWork()) {
                std::uint64_t rows=0;
                for(int request=extent.projectionWork.requestForRow(0);request<=extent.projectionWork.requestForRow(extent.projectionWork.physicalRows()-1);++request)
                    rows+=static_cast<unsigned>(extent.projectionWork.resolveDevice(request).size());
                result->total_checked=static_cast<unsigned>(rows*extent.columns);
                return;
            }
            auto geometry = extent.geometry;
            std::int32_t prior = 0, live = 0;
            if (geometry.kind == SnapshotContiguousRowKind::StatePrefixRows)
            {
                prior = extent.frontier[0];
                live = extent.frontier[1];
            }
            else if (geometry.kind == SnapshotContiguousRowKind::TokenRows && extent.frontier)
                live = extent.frontier[0];
            else
                geometry.kind = SnapshotContiguousRowKind::Physical;
            const auto observed = resolveSnapshotContiguousRows(geometry, prior, live);
            result->invalid_extent = observed.valid ? 0U : 1U;
            result->total_checked = observed.valid
                ? static_cast<unsigned int>(observed.rows * extent.columns) : 0U;
        }
    }

    /** Atomic float minimum with correct ordering for negative values. */
    __device__ void atomicMinFloatROCm(float *address, float value)
    {
        int *bits = reinterpret_cast<int *>(address);
        int observed = *bits;
        while (value < __int_as_float(observed))
        {
            const int assumed = observed;
            observed = atomicCAS(bits, assumed, __float_as_int(value));
            if (observed == assumed)
                break;
        }
    }

    /** Atomic float maximum with correct ordering for negative values. */
    __device__ void atomicMaxFloatROCm(float *address, float value)
    {
        int *bits = reinterpret_cast<int *>(address);
        int observed = *bits;
        while (value > __int_as_float(observed))
        {
            const int assumed = observed;
            observed = atomicCAS(bits, assumed, __float_as_int(value));
            if (observed == assumed)
                break;
        }
    }

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
        DeviceValidationResult *__restrict__ result, TensorValidationExtent extent)
    {
        // Initialization on this stream has already authenticated the frontier.
        num_elements = extent.usesProjectionWork() ? num_elements : result->total_checked;
        if(result->invalid_extent)return;

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
        bool local_nonzero = false;
        float local_min = std::numeric_limits<float>::max();
        float local_max = std::numeric_limits<float>::lowest();

        // Grid-stride loop
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < num_elements; i += stride)
        {
            if(extent.usesProjectionWork()&&!extent.projectionWork.rowIsActive(static_cast<int>(i/extent.columns)))continue;
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
                else
                {
                    local_nonzero = true;
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
        if (local_nonzero)
            atomicOr(&s_has_nonzero, 1u);

        // Use atomicMin/Max for floats (reinterpret as int for ordered comparison)
        // This works because IEEE floats have the property that their bit patterns
        // sort correctly when interpreted as signed integers (for positive values)
        if (local_min < std::numeric_limits<float>::max())
        {
            atomicMinFloatROCm(&s_min, local_min);
        }
        if (local_max > std::numeric_limits<float>::lowest())
        {
            atomicMaxFloatROCm(&s_max, local_max);
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
            atomicMinFloatROCm(&result->sample_min, s_min);
            atomicMaxFloatROCm(&result->sample_max, s_max);
        }
    }

    /**
     * @brief BF16 validation kernel
     */
    __global__ void validateBF16Kernel(
        const uint16_t *__restrict__ data,
        size_t num_elements,
        DeviceValidationResult *__restrict__ result, TensorValidationExtent extent)
    {
        // Initialization on this stream has already authenticated the frontier.
        num_elements = extent.usesProjectionWork() ? num_elements : result->total_checked;
        if(result->invalid_extent)return;

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
        bool local_nonzero = false;

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < num_elements; i += stride)
        {
            if(extent.usesProjectionWork()&&!extent.projectionWork.rowIsActive(static_cast<int>(i/extent.columns)))continue;
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
            else
            {
                local_nonzero = true;
            }
        }

        if (local_nan > 0)
            atomicAdd(&s_nan_count, local_nan);
        if (local_inf > 0)
            atomicAdd(&s_inf_count, local_inf);
        if (local_zero > 0)
            atomicAdd(&s_zero_count, local_zero);
        if (local_nonzero)
            atomicOr(&s_has_nonzero, 1u);

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
        DeviceValidationResult *__restrict__ result, TensorValidationExtent extent)
    {
        // Initialization on this stream has already authenticated the frontier.
        num_elements = extent.usesProjectionWork() ? num_elements : result->total_checked;
        if(result->invalid_extent)return;

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
        bool local_nonzero = false;

        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        size_t stride = blockDim.x * gridDim.x;

        for (size_t i = idx; i < num_elements; i += stride)
        {
            if(extent.usesProjectionWork()&&!extent.projectionWork.rowIsActive(static_cast<int>(i/extent.columns)))continue;
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
            else
            {
                local_nonzero = true;
            }
        }

        if (local_nan > 0)
            atomicAdd(&s_nan_count, local_nan);
        if (local_inf > 0)
            atomicAdd(&s_inf_count, local_inf);
        if (local_zero > 0)
            atomicAdd(&s_zero_count, local_zero);
        if (local_nonzero)
            atomicOr(&s_has_nonzero, 1u);

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
            requireSuccess(hipSetDevice(device_id_), "hipSetDevice during construction");
            requireSuccess(
                hipMalloc(&d_result_, sizeof(DeviceValidationResult)),
                "hipMalloc persistent device result");

            hipError_t host_error = hipHostMalloc(
                reinterpret_cast<void **>(&h_result_),
                sizeof(DeviceValidationResult));
            if (host_error != hipSuccess)
            {
                (void)hipFree(d_result_);
                d_result_ = nullptr;
                requireSuccess(host_error, "hipHostMalloc persistent host result");
            }

            hipError_t event_error = hipEventCreateWithFlags(
                &completion_event_,
                hipEventDisableTiming);
            if (event_error != hipSuccess)
            {
                (void)hipHostFree(h_result_);
                h_result_ = nullptr;
                (void)hipFree(d_result_);
                d_result_ = nullptr;
                requireSuccess(event_error, "hipEventCreateWithFlags completion event");
            }
        }

        ~ROCmTensorValidator() override
        {
            (void)hipSetDevice(device_id_);
            if (completion_event_)
                (void)hipEventDestroy(completion_event_);
            if (h_result_)
                (void)hipHostFree(h_result_);
            if (d_result_)
                (void)hipFree(d_result_);
        }

        /** @copydoc ITensorValidator::validate */
        [[nodiscard]] TensorValidationResult validate(
            const void *device_ptr,
            size_t num_elements,
            TensorValidationDataType data_type,
            ExplicitGPUStream producer_stream) override
        {
            return validateObserved(device_ptr,
                {{SnapshotContiguousRowKind::Physical, num_elements}, 1, nullptr},
                data_type, producer_stream);
        }

        /** @copydoc ITensorValidator::validateObserved */
        [[nodiscard]] TensorValidationResult validateObserved(
            const void *device_ptr,
            TensorValidationExtent extent,
            TensorValidationDataType data_type,
            ExplicitGPUStream producer_stream) override
        {
            if (!device_ptr)
                throw std::invalid_argument("ROCmTensorValidator requires a non-null device pointer");
            const auto num_elements = checkedTensorValidationElementCapacity(extent, data_type);

            std::lock_guard<std::mutex> invocation_lock(invocation_mutex_);
            requireSuccess(hipSetDevice(device_id_), "hipSetDevice before validation");
            auto stream = reinterpret_cast<hipStream_t>(producer_stream.get());
            const int block_size = 256;
            const int max_blocks = 1024;
            int num_blocks = static_cast<int>(std::min(static_cast<size_t>(max_blocks), (num_elements + block_size - 1) / block_size));

            hipLaunchKernelGGL(
                initializeValidationResultROCm,
                dim3(1),
                dim3(1),
                0,
                stream,
                d_result_,
                extent);
            requireSuccess(hipGetLastError(), "initialize validation result launch");

            switch (data_type)
            {
            case TensorValidationDataType::FP32:
                hipLaunchKernelGGL(
                    validateFP32Kernel,
                    dim3(num_blocks),
                    dim3(block_size),
                    0,
                    stream,
                    static_cast<const float *>(device_ptr),
                    num_elements,
                    d_result_, extent);
                break;
            case TensorValidationDataType::BF16:
                hipLaunchKernelGGL(
                    validateBF16Kernel,
                    dim3(num_blocks),
                    dim3(block_size),
                    0,
                    stream,
                    static_cast<const uint16_t *>(device_ptr),
                    num_elements,
                    d_result_, extent);
                break;
            case TensorValidationDataType::FP16:
                hipLaunchKernelGGL(
                    validateFP16Kernel,
                    dim3(num_blocks),
                    dim3(block_size),
                    0,
                    stream,
                    static_cast<const uint16_t *>(device_ptr),
                    num_elements,
                    d_result_, extent);
                break;
            }
            requireSuccess(hipGetLastError(), "tensor validation kernel launch");

            requireSuccess(
                hipMemcpyAsync(
                    h_result_,
                    d_result_,
                    sizeof(DeviceValidationResult),
                    hipMemcpyDeviceToHost,
                    stream),
                "compact validation result D2H");
            requireSuccess(
                hipEventRecord(completion_event_, stream),
                "validation completion event publication");
            requireSuccess(
                hipEventSynchronize(completion_event_),
                "validation completion event wait");

            TensorValidationResult result;
            result.has_nan = (h_result_->has_nan != 0);
            result.has_inf = (h_result_->has_inf != 0);
            result.appears_zero = (h_result_->appears_zero != 0);
            result.invalid_extent = (h_result_->invalid_extent != 0);
            result.valid = !result.invalid_extent && !result.has_nan && !result.has_inf;
            result.nan_count = h_result_->nan_count;
            result.inf_count = h_result_->inf_count;
            result.zero_count = h_result_->zero_count;
            result.total_checked = h_result_->total_checked;
            result.sample_min = h_result_->sample_min;
            result.sample_max = h_result_->sample_max;
            return result;
        }

    private:
        static void requireSuccess(hipError_t error, const char *operation)
        {
            if (error == hipSuccess)
                return;
            throw std::runtime_error(
                std::string("ROCmTensorValidator ") + operation + " failed: " +
                hipGetErrorString(error));
        }

        DeviceValidationResult *d_result_ = nullptr;
        DeviceValidationResult *h_result_ = nullptr;
        hipEvent_t completion_event_ = nullptr;
        int device_id_ = -1;
        std::mutex invocation_mutex_;
    };

    // =========================================================================
    // Per-Device Validator Factory (Thread-Safe)
    // =========================================================================

    static std::mutex g_rocm_validator_mutex;
    static std::unordered_map<int, std::unique_ptr<ROCmTensorValidator>> g_rocm_validators;

    ITensorValidator *getROCmTensorValidator(int device_id)
    {
        if (device_id < 0)
        {
            LOG_ERROR("[getROCmTensorValidator] Invalid device ordinal " << device_id);
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(g_rocm_validator_mutex);

        auto it = g_rocm_validators.find(device_id);
        if (it == g_rocm_validators.end())
        {
            auto validator = std::make_unique<ROCmTensorValidator>(device_id);
            auto *ptr = validator.get();
            g_rocm_validators[device_id] = std::move(validator);
            LOG_DEBUG("[getROCmTensorValidator] Created validator for device " << device_id);
            return ptr;
        }
        return it->second.get();
    }

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

        return getROCmTensorValidator(device_id);
    }

    /**
     * @brief Destroy one validator before its HIP runtime generation resets.
     *
     * Registry publication is removed before destructor work enters HIP, so no
     * caller can acquire a validator whose device allocation or completion
     * event is in the process of retirement.
     */
    bool retireROCmTensorValidatorRuntimeGeneration(int device_id)
    {
        std::unique_ptr<ROCmTensorValidator> retired;
        {
            std::lock_guard<std::mutex> lock(g_rocm_validator_mutex);
            const auto iterator = g_rocm_validators.find(device_id);
            if (iterator == g_rocm_validators.end())
                return true;
            retired = std::move(iterator->second);
            g_rocm_validators.erase(iterator);
        }
        retired.reset();
        return true;
    }

} // namespace llaminar2

// C linkage export for cross-TU factory
extern "C" llaminar2::ITensorValidator *llaminar2_getROCmTensorValidator()
{
    return llaminar2::getROCmTensorValidator();
}

extern "C" llaminar2::ITensorValidator *llaminar2_getROCmTensorValidatorForDevice(int device_id)
{
    return llaminar2::getROCmTensorValidator(device_id);
}

/** @brief C-linkage retirement hook owned by ROCm backend lifecycle code. */
extern "C" bool llaminar2_retireROCmTensorValidatorRuntimeGeneration(
    int device_id)
{
    return llaminar2::retireROCmTensorValidatorRuntimeGeneration(device_id);
}
