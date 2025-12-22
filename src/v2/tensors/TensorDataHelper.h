/**
 * @file TensorDataHelper.h
 * @brief Safe tensor data extraction utilities with fail-fast validation
 * @author David Sanftenberg
 * @date December 2025
 *
 * This file provides helper functions for safely extracting data from tensors,
 * particularly when dealing with polymorphic tensor types (FP32Tensor, TensorSlice,
 * quantized tensors). These helpers prevent the silent failure modes that led
 * to tensor parallelism debugging issues.
 *
 * **Key Principle**: Never use dynamic_cast<FP32Tensor*> when you just need FP32 data.
 * Use the unified TensorBase::fp32_data() / TensorBase::data() interface instead.
 *
 * @see changelog/2025-12-22-tensor-parallelism-debugging-retrospective.md
 */

#pragma once

#include "Tensors.h"
#include "../utils/Logger.h"
#include <stdexcept>
#include <string>
#include <cassert>

namespace llaminar2
{

    /**
     * @brief Helper utilities for safe tensor data extraction
     *
     * These functions provide fail-fast extraction of tensor data with clear
     * error messages. They are designed to catch bugs early rather than
     * silently returning nullptr.
     */
    class TensorDataHelper
    {
    public:
        // =========================================================================
        // FP32 Bias Extraction (Most Common Use Case)
        // =========================================================================

        /**
         * @brief Extract FP32 bias data from tensor, with optional validation
         *
         * Safely extracts FP32 data pointer from any tensor type that supports it:
         * - FP32Tensor: Direct data() call
         * - TensorSlice: Delegates to inner tensor's data()
         * - Quantized tensors: NOT supported (returns nullptr unless require_data is false)
         *
         * @param tensor Tensor to extract bias from (may be nullptr)
         * @param context_name Name for error messages (e.g., "q_bias", "layer5_v_bias")
         * @param require_data If true and tensor is non-null but data extraction fails, throws
         * @return const float* Pointer to FP32 data, or nullptr if tensor is nullptr
         *
         * @throws std::runtime_error if require_data is true and extraction fails
         *
         * **Usage**:
         * @code
         * // Optional bias (common case)
         * const float* q_bias = TensorDataHelper::extractBias(layer.q_bias, "q_bias");
         *
         * // Required bias (throws if extraction fails)
         * const float* q_bias = TensorDataHelper::extractBias(layer.q_bias, "q_bias", true);
         * @endcode
         */
        static const float *extractBias(
            const TensorBase *tensor,
            const std::string &context_name,
            bool require_data = false)
        {
            if (!tensor)
            {
                return nullptr;
            }

            // Use unified interface - works for FP32Tensor, TensorSlice, etc.
            const float *data = tensor->data();

            if (!data && require_data)
            {
                throw std::runtime_error(
                    "TensorDataHelper::extractBias: Failed to extract FP32 data from '" +
                    context_name + "' (tensor type: " + tensor->dtype_name() + ")");
            }

            return data;
        }

        /**
         * @brief Extract FP32 bias data from unique_ptr
         */
        static const float *extractBias(
            const std::unique_ptr<TensorBase> &tensor,
            const std::string &context_name,
            bool require_data = false)
        {
            return extractBias(tensor.get(), context_name, require_data);
        }

        /**
         * @brief Extract FP32 bias data from shared_ptr
         */
        static const float *extractBias(
            const std::shared_ptr<TensorBase> &tensor,
            const std::string &context_name,
            bool require_data = false)
        {
            return extractBias(tensor.get(), context_name, require_data);
        }

        // =========================================================================
        // Validation Utilities
        // =========================================================================

        /**
         * @brief Validate tensor shape matches expected dimensions
         *
         * @param tensor Tensor to validate
         * @param expected_shape Expected shape vector
         * @param context_name Name for error messages
         * @throws std::runtime_error if shapes don't match
         */
        static void validateShape(
            const TensorBase *tensor,
            const std::vector<size_t> &expected_shape,
            const std::string &context_name)
        {
            if (!tensor)
            {
                throw std::runtime_error(
                    "TensorDataHelper::validateShape: '" + context_name + "' is nullptr");
            }

            const auto &actual = tensor->shape();
            if (actual.size() != expected_shape.size())
            {
                throw std::runtime_error(
                    "TensorDataHelper::validateShape: '" + context_name +
                    "' rank mismatch. Expected " + std::to_string(expected_shape.size()) +
                    " dims, got " + std::to_string(actual.size()));
            }

            for (size_t i = 0; i < actual.size(); ++i)
            {
                if (actual[i] != expected_shape[i])
                {
                    throw std::runtime_error(
                        "TensorDataHelper::validateShape: '" + context_name +
                        "' dimension " + std::to_string(i) + " mismatch. Expected " +
                        std::to_string(expected_shape[i]) + ", got " + std::to_string(actual[i]));
                }
            }
        }

        /**
         * @brief Check if tensor is FP32-backed (zero-cost data access)
         *
         * @param tensor Tensor to check
         * @return true if fp32_data() is zero-cost (no conversion needed)
         */
        static bool isFP32Backed(const TensorBase *tensor)
        {
            return tensor && tensor->is_fp32_backed();
        }

        /**
         * @brief Check if tensor is a view/wrapper (like TensorSlice)
         *
         * @param tensor Tensor to check
         * @return true if this is a view/slice of another tensor
         */
        static bool isView(const TensorBase *tensor)
        {
            return tensor && tensor->is_view();
        }

        // =========================================================================
        // Debug Utilities
        // =========================================================================

        /**
         * @brief Get detailed type info for debugging
         *
         * @param tensor Tensor to describe
         * @return String like "FP32Tensor (view)" or "TensorSlice<Q4_0>"
         */
        static std::string describeType(const TensorBase *tensor)
        {
            if (!tensor)
            {
                return "nullptr";
            }

            std::string desc = tensor->dtype_name();
            if (tensor->is_view())
            {
                desc += " (view)";
            }
            if (tensor->is_fp32_backed())
            {
                desc += " [FP32-backed]";
            }
            return desc;
        }

#ifndef NDEBUG
        /**
         * @brief Debug assertion that tensor can provide FP32 data
         *
         * Only active in debug builds. Logs detailed info on failure.
         */
        static void assertCanProvideFP32Data(
            const TensorBase *tensor,
            const std::string &context_name)
        {
            if (!tensor)
            {
                return; // nullptr is valid (optional tensor)
            }

            const float *data = tensor->data();
            if (!data)
            {
                LOG_ERROR("TensorDataHelper assertion failed: '" << context_name
                                                                 << "' cannot provide FP32 data. Type: " << describeType(tensor));
                assert(false && "Tensor cannot provide FP32 data");
            }
        }
#else
        // No-op in release
        static void assertCanProvideFP32Data(const TensorBase *, const std::string &) {}
#endif
    };

} // namespace llaminar2
