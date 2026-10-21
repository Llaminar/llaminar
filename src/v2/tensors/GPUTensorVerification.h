/**
 * @file GPUTensorVerification.h
 * @brief Stream-ordered GPU tensor validation (NaN/Inf/zero detection).
 *
 * Provides device-side tensor validation for Debug/Integration builds without
 * materializing tensor payloads on the host. The validator consumes the exact
 * producer stream, performs the scan on that stream, and transfers only its
 * small diagnostic result after an event has published completion.
 *
 * @author David Sanftenberg
 *
 * @see TensorVerification.h (related host-side validation)
 */

#pragma once

#include "../snapshots/SnapshotContiguousRows.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include "../backends/DeviceType.h"
#include "../backends/ExplicitGPUStream.h"

namespace llaminar2
{
    /**
     * @brief Result of GPU-side tensor validation
     *
     * Small struct transferred D2H after validation kernel completes.
     * The completed record fits in a single cache line; no separate source-count
     * readback or payload download is added.
     */
    struct TensorValidationResult
    {
        bool has_nan = false;     ///< True if any NaN values detected
        bool has_inf = false;     ///< True if any Inf values detected
        bool appears_zero = true; ///< True if all sampled values are zero
        bool valid = true;        ///< Valid acquired extent and no NaN/Inf
        bool invalid_extent = false; ///< Malformed frontier; never an authenticated empty result

        uint32_t nan_count = 0;     ///< Number of NaN values found (capped at UINT32_MAX)
        uint32_t inf_count = 0;     ///< Number of Inf values found (capped at UINT32_MAX)
        uint32_t zero_count = 0;    ///< Number of zero values found
        uint32_t total_checked = 0; ///< Total elements checked

        float sample_min = 0.0f; ///< Minimum non-NaN value
        float sample_max = 0.0f; ///< Maximum non-NaN value

        /// Check if tensor passes validation
        bool passed() const { return valid && !has_nan && !has_inf; }
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
     * @brief Borrow of the original native frontier for one validation transaction.
     *
     * Physical geometry has no frontier. Token rows optionally borrow one INT32
     * live count (absence means fully active); state prefixes require two
     * adjacent INT32 words. The stage authenticates original tensor ownership,
     * subview bounds and lifetime before constructing this call-only descriptor.
     * The initializer acquires these words on the exact producer stream.
     */
    struct TensorValidationExtent
    {
        SnapshotContiguousRowGeometry geometry;
        std::size_t columns = 1;
        const std::int32_t *frontier = nullptr;
    };

    /**
     * @brief Check static observation geometry before backend work or allocation.
     * @param extent Frozen geometry and call-only original frontier address.
     * @param data_type Actual floating representation submitted to the scanner.
     * @return Positive physical element capacity, bounded by exact uint32 counters.
     * @throws std::invalid_argument for overflow, unknown axes/representations,
     *         missing state frontiers or misaligned/malformed static bindings.
     *
     * This function never dereferences the frontier. CPU Unit tests can validate
     * counter boundaries without constructing a backend or allocating a tensor.
     */
    [[nodiscard]] inline std::size_t checkedTensorValidationElementCapacity(
        TensorValidationExtent extent, TensorValidationDataType data_type)
    {
        const auto &geometry = extent.geometry;
        if (!geometry.capacity || !extent.columns ||
            geometry.capacity > UINT32_MAX / extent.columns)
            throw std::invalid_argument("Tensor validation diagnostic extent exceeds positive UINT32 counters");
        if (geometry.kind != SnapshotContiguousRowKind::Physical &&
            geometry.kind != SnapshotContiguousRowKind::TokenRows &&
            geometry.kind != SnapshotContiguousRowKind::StatePrefixRows)
            throw std::invalid_argument("Tensor validation has an unknown observation axis");
        if ((geometry.kind == SnapshotContiguousRowKind::Physical && extent.frontier) ||
            (extent.frontier && reinterpret_cast<std::uintptr_t>(extent.frontier) % alignof(std::int32_t)) ||
            (geometry.kind == SnapshotContiguousRowKind::StatePrefixRows &&
             (!extent.frontier || geometry.sourceCapacity <= 0 || geometry.sourceRowsPerStateRow <= 0)))
            throw std::invalid_argument("Tensor validation has an invalid original frontier binding");
        if (data_type != TensorValidationDataType::FP32 &&
            data_type != TensorValidationDataType::BF16 && data_type != TensorValidationDataType::FP16)
            throw std::invalid_argument("Tensor validation has an unknown floating representation");
        return geometry.capacity * extent.columns;
    }

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

        /**
         * @brief Validate only authored contiguous rows, acquiring counts natively.
         * @param device_ptr Original non-null, bounded output subview.
         * @param extent Frozen physical geometry and authenticated native frontier.
         * @param data_type Exact native floating-point representation.
         * @param producer_stream Exact stream producing both payload and frontier.
         * @return Existing compact result with actual total_checked, including
         *         valid-empty or distinct invalid_extent; no count-only readback.
         * @throws std::invalid_argument for malformed static geometry/pointers or
         *         diagnostic capacity above UINT32_MAX elements (counter width).
         * @throws std::runtime_error for infrastructure/launch/publication failure.
         */
        [[nodiscard]] virtual TensorValidationResult validateObserved(
            const void *device_ptr,
            TensorValidationExtent extent,
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
