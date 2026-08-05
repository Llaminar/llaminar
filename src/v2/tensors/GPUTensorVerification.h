/**
 * @file GPUTensorVerification.h
 * @brief Stream-ordered GPU tensor validation (NaN/Inf/zero detection).
 *
 * Provides device-side tensor validation for Debug/Integration builds without
 * materializing tensor payloads on the host. The validator consumes the exact
 * producer stream, performs the scan on that stream, and transfers only its
 * small diagnostic result after an event has published completion.
 *
 * Performance impact:
 * - Traditional: 10+ seconds per large tensor (hipMemcpy D2H + host scan)
 * - GPU-side: ~1ms per tensor (kernel launch + 32 byte D2H)
 *
 * @author David Sanftenberg
 *
 * @see TensorVerification.h (related host-side validation)
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include "../backends/DeviceType.h"
#include "../backends/ExplicitGPUStream.h"

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
     * @brief Floating-point storage representations supported by GPU validation.
     *
     * Keeping the representation typed prevents string parsing and accidental
     * reinterpretation after the stage verifier has selected a device pointer.
     */
    enum class TensorValidationDataType : uint8_t
    {
        FP32,
        BF16,
        FP16,
    };

    /**
     * @brief GPU-side tensor validation interface
     *
     * Implemented by CUDA and ROCm backends to provide device-side NaN/Inf and
     * exact all-zero checking. One call is one complete diagnostic transaction:
     * initialization, scan, result publication, and the small terminal D2H all
     * remain ordered on the supplied producer stream. Implementations throw on
     * infrastructure or launch failure; they never substitute a default stream
     * or request a host copy of the source tensor.
     */
    class ITensorValidator
    {
    public:
        virtual ~ITensorValidator() = default;

        /**
         * @brief Validate one floating-point tensor on its producer stream.
         *
         * @param device_ptr Non-null pointer to tensor storage on this
         *        validator's device.
         * @param num_elements Positive number of logical elements to scan.
         * @param data_type Physical floating-point representation.
         * @param producer_stream Exact non-null stream that produced the tensor.
         * @return Completed validation result after only the compact diagnostic
         *         record has crossed to the host.
         * @throws std::runtime_error if validation cannot complete exactly as
         *         requested.
         */
        [[nodiscard]] virtual TensorValidationResult validate(
            const void *device_ptr,
            size_t num_elements,
            TensorValidationDataType data_type,
            ExplicitGPUStream producer_stream) = 0;
    };

    /**
     * @brief Get the tensor validator for a device type
     *
     * @param device_type DeviceType::CUDA or DeviceType::ROCM
     * @return Pointer to validator, or nullptr if not available
     */
    ITensorValidator *getTensorValidator(DeviceType device_type);
    ITensorValidator *getTensorValidator(DeviceType device_type, int device_id);

    // Backend-specific factory functions (defined in CUDA/ROCm compilation units)
    ITensorValidator *getCUDATensorValidator();
    ITensorValidator *getCUDATensorValidator(int device_id);
    ITensorValidator *getROCmTensorValidator();
    ITensorValidator *getROCmTensorValidator(int device_id);

} // namespace llaminar2
