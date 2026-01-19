/**
 * @file TensorValidation.h
 * @brief GPU-accelerated tensor validation (NaN/Inf/Zero detection)
 *
 * Provides device-side tensor validation to avoid expensive D2H transfers
 * during Debug/Integration builds. Instead of copying entire tensors to host,
 * we run a reduction kernel on GPU and transfer only the validation result.
 *
 * Performance impact:
 * - Traditional: 10+ seconds per large tensor (hipMemcpy D2H + host scan)
 * - GPU-side: ~1ms per tensor (kernel launch + 32 byte D2H)
 *
 * @author David Sanftenberg
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "../backends/DeviceType.h"

namespace llaminar2
{
    /**
     * @brief Result of GPU-side tensor validation
     *
     * Small struct transferred D2H after validation kernel completes.
     * Total size: 32 bytes (fits in a single cache line)
     */
    struct TensorValidationResult
    {
        bool has_nan = false;     ///< True if any NaN values detected
        bool has_inf = false;     ///< True if any Inf values detected
        bool appears_zero = true; ///< True if all sampled values are zero
        bool valid = true;        ///< Overall validity (no NaN/Inf)

        uint32_t nan_count = 0;     ///< Number of NaN values found (capped at UINT32_MAX)
        uint32_t inf_count = 0;     ///< Number of Inf values found (capped at UINT32_MAX)
        uint32_t zero_count = 0;    ///< Number of zero values found
        uint32_t total_checked = 0; ///< Total elements checked

        float sample_min = 0.0f; ///< Minimum non-NaN value
        float sample_max = 0.0f; ///< Maximum non-NaN value

        /// Check if tensor passes validation
        bool passed() const { return !has_nan && !has_inf; }
    };

    /**
     * @brief GPU-side tensor validation interface
     *
     * Implemented by CUDA and ROCm backends to provide device-side NaN/Inf checking.
     */
    class ITensorValidator
    {
    public:
        virtual ~ITensorValidator() = default;

        /**
         * @brief Validate FP32 tensor on GPU
         *
         * Launches a reduction kernel to scan for NaN/Inf/zero values.
         * Returns immediately after kernel launch; call getResult() to
         * synchronize and retrieve the result.
         *
         * @param device_ptr Pointer to FP32 data on GPU
         * @param num_elements Number of float elements to check
         * @param device_id GPU device ordinal
         * @return true if validation kernel launched successfully
         */
        virtual bool validateFP32Async(const void *device_ptr,
                                       size_t num_elements,
                                       int device_id) = 0;

        /**
         * @brief Validate BF16 tensor on GPU
         *
         * @param device_ptr Pointer to BF16 data on GPU (uint16_t*)
         * @param num_elements Number of BF16 elements to check
         * @param device_id GPU device ordinal
         * @return true if validation kernel launched successfully
         */
        virtual bool validateBF16Async(const void *device_ptr,
                                       size_t num_elements,
                                       int device_id) = 0;

        /**
         * @brief Validate FP16 tensor on GPU
         *
         * @param device_ptr Pointer to FP16 data on GPU (uint16_t*)
         * @param num_elements Number of FP16 elements to check
         * @param device_id GPU device ordinal
         * @return true if validation kernel launched successfully
         */
        virtual bool validateFP16Async(const void *device_ptr,
                                       size_t num_elements,
                                       int device_id) = 0;

        /**
         * @brief Get validation result (synchronizes with GPU)
         *
         * Waits for the validation kernel to complete and transfers
         * the result struct from GPU to host (~32 bytes).
         *
         * @param result Output validation result
         * @return true if result was successfully retrieved
         */
        virtual bool getResult(TensorValidationResult &result) = 0;

        /**
         * @brief Synchronous validation (launch + wait + get result)
         *
         * Convenience method that combines validateFP32Async + getResult.
         *
         * @param device_ptr Pointer to FP32 data on GPU
         * @param num_elements Number of float elements to check
         * @param device_id GPU device ordinal
         * @param result Output validation result
         * @return true if validation completed successfully
         */
        virtual bool validateFP32(const void *device_ptr,
                                  size_t num_elements,
                                  int device_id,
                                  TensorValidationResult &result)
        {
            if (!validateFP32Async(device_ptr, num_elements, device_id))
                return false;
            return getResult(result);
        }
    };

    /**
     * @brief Get the tensor validator for a device type
     *
     * @param device_type DeviceType::CUDA or DeviceType::ROCM
     * @return Pointer to validator, or nullptr if not available
     */
    ITensorValidator *getTensorValidator(DeviceType device_type);

    // Backend-specific factory functions (defined in CUDA/ROCm compilation units)
    ITensorValidator *getCUDATensorValidator();
    ITensorValidator *getROCmTensorValidator();

} // namespace llaminar2
