/**
 * @file TensorValidation.h
 * @brief Debug-only tensor validation utilities
 * @author GitHub Copilot
 * @date December 2025
 *
 * Provides utilities to detect uninitialized, zero, or corrupted tensors
 * in debug builds. These checks are compiled out in Release mode.
 *
 * ## Why This Exists
 *
 * During Hybrid mode debugging, we discovered that `V_dequant` was allocated
 * but never populated, causing attention output to be all zeros. This bug
 * went undetected because the attention kernel silently accepted zeros.
 *
 * ## Usage
 *
 * @code
 * #ifndef NDEBUG
 * if (tensorAppearsZero(V_tensor, 1000)) {
 *     LOG_WARN("V tensor appears uninitialized!");
 * }
 * validateTensorNotZero(V_tensor, "V", "AttentionCompute");
 * #endif
 * @endcode
 */

#pragma once

#include "Tensors.h"
#include "../utils/Logger.h"

#include <cmath>
#include <string>
#include <algorithm>

namespace llaminar2
{

#ifndef NDEBUG

    /**
     * @brief Check if a tensor appears to be uninitialized (all zeros)
     *
     * Samples up to `sample_count` elements from the tensor and checks if
     * they're all zero. For large tensors, sampling is more efficient than
     * checking every element.
     *
     * @param tensor Tensor to check (can be null)
     * @param sample_count Maximum elements to sample (default: 1000)
     * @return true if tensor is null or appears to be all zeros
     */
    inline bool tensorAppearsZero(const TensorBase *tensor, size_t sample_count = 1000)
    {
        if (!tensor)
            return true;

        size_t numel = tensor->numel();
        if (numel == 0)
            return true;

        // Check FP32 tensors
        if (auto *fp32 = dynamic_cast<const FP32Tensor *>(tensor))
        {
            const float *data = fp32->data();
            if (!data)
                return true;

            size_t check_count = std::min(numel, sample_count);
            size_t stride = (numel > sample_count) ? numel / sample_count : 1;

            for (size_t i = 0; i < check_count; ++i)
            {
                size_t idx = (i * stride) % numel;
                if (data[idx] != 0.0f)
                    return false;
            }
            return true;
        }

        // For other tensor types (BF16, quantized), assume valid
        // The primary use case is FP32 buffers like V_dequant
        return false;
    }

    /**
     * @brief Check if tensor contains NaN or Inf values
     *
     * @param tensor Tensor to check (can be null)
     * @param sample_count Maximum elements to sample (default: 1000)
     * @return true if tensor contains NaN or Inf
     */
    inline bool tensorHasNaNOrInf(const TensorBase *tensor, size_t sample_count = 1000)
    {
        if (!tensor)
            return false;

        size_t numel = tensor->numel();
        if (numel == 0)
            return false;

        if (auto *fp32 = dynamic_cast<const FP32Tensor *>(tensor))
        {
            const float *data = fp32->data();
            if (!data)
                return false;

            size_t check_count = std::min(numel, sample_count);
            size_t stride = (numel > sample_count) ? numel / sample_count : 1;

            for (size_t i = 0; i < check_count; ++i)
            {
                size_t idx = (i * stride) % numel;
                if (std::isnan(data[idx]) || std::isinf(data[idx]))
                    return true;
            }
        }

        // For other tensor types, assume no NaN/Inf
        return false;
    }

    /**
     * @brief Validate tensor and log warning if problematic
     *
     * Combines zero and NaN/Inf checks with contextual logging.
     *
     * @param tensor Tensor to validate
     * @param tensor_name Name for logging (e.g., "V", "K_rope")
     * @param stage_name Stage name for context (e.g., "AttentionCompute")
     * @param sample_count Maximum elements to sample
     */
    inline void validateTensorNotZero(const TensorBase *tensor,
                                      const std::string &tensor_name,
                                      const std::string &stage_name,
                                      size_t sample_count = 1000)
    {
        if (!tensor)
        {
            LOG_WARN("[" << stage_name << "] Tensor '" << tensor_name << "' is null!");
            return;
        }

        if (tensorAppearsZero(tensor, sample_count))
        {
            LOG_WARN("[" << stage_name << "] Tensor '" << tensor_name
                         << "' appears to be all zeros (likely uninitialized)!"
                         << " numel=" << tensor->numel());
        }

        if (tensorHasNaNOrInf(tensor, sample_count))
        {
            LOG_WARN("[" << stage_name << "] Tensor '" << tensor_name
                         << "' contains NaN or Inf values!"
                         << " numel=" << tensor->numel());
        }
    }

    /**
     * @brief Assert tensor is valid (non-null, non-zero, no NaN/Inf)
     *
     * Unlike validateTensorNotZero which logs warnings, this throws
     * if validation fails. Use for critical invariants.
     *
     * @param tensor Tensor to validate
     * @param tensor_name Name for error message
     * @param stage_name Stage name for context
     * @throws std::runtime_error if validation fails
     */
    inline void assertTensorValid(const TensorBase *tensor,
                                  const std::string &tensor_name,
                                  const std::string &stage_name)
    {
        if (!tensor)
        {
            throw std::runtime_error("[" + stage_name + "] Tensor '" + tensor_name + "' is null!");
        }

        if (tensorAppearsZero(tensor))
        {
            throw std::runtime_error("[" + stage_name + "] Tensor '" + tensor_name +
                                     "' appears to be all zeros (likely uninitialized)! numel=" +
                                     std::to_string(tensor->numel()));
        }

        if (tensorHasNaNOrInf(tensor))
        {
            throw std::runtime_error("[" + stage_name + "] Tensor '" + tensor_name +
                                     "' contains NaN or Inf values!");
        }
    }

#else // NDEBUG - Release mode: compile out validation

    inline bool tensorAppearsZero(const TensorBase *, size_t = 1000) { return false; }
    inline bool tensorHasNaNOrInf(const TensorBase *, size_t = 1000) { return false; }
    inline void validateTensorNotZero(const TensorBase *, const std::string &,
                                      const std::string &, size_t = 1000) {}
    inline void assertTensorValid(const TensorBase *, const std::string &,
                                  const std::string &) {}

#endif // NDEBUG

} // namespace llaminar2
