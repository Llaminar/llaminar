/**
 * @file TensorDimensions.h
 * @brief Type-safe tensor dimension validation for pipeline stages
 *
 * Provides compile-time and runtime dimension checking to prevent
 * accidental transpositions and shape mismatches between pipeline stages.
 *
 * Features:
 * - Zero overhead in Release builds (compiles to no-ops via NDEBUG)
 * - Strong validation in Debug builds (aborts with helpful error messages)
 * - Pipeline-agnostic design (each pipeline defines its own dimensions)
 * - Clear error messages with stage context
 *
 * Design Philosophy:
 * - Pipelines define helper methods to create TensorSpecs for their architecture
 * - Validation calls sprinkled throughout forward() pass to catch bugs early
 * - No performance cost in production (Release builds)
 *
 * Usage Pattern (in pipeline implementation):
 *
 *   // 1. Define architecture-specific spec helpers (in pipeline class):
 *   TensorSpec spec_hidden(int seq_len) const {
 *       return TensorSpec({seq_len, d_model_},
 *                         "hidden[" + std::to_string(seq_len) + "," +
 *                         std::to_string(d_model_) + "]");
 *   }
 *
 *   // 2. Validate tensors at each pipeline stage:
 *   VALIDATE_TENSOR(current_hidden_, spec_hidden(seq_len), "after_embedding");
 *   VALIDATE_TENSOR(Q, spec_q(seq_len), "after_q_proj");
 *   VALIDATE_TENSOR(gate, spec_ffn_intermediate(seq_len), "after_gate_proj");
 *
 *   // 3. Validate same-shape requirements:
 *   ASSERT_SAME_SHAPE(gate, up, "ffn_swiglu_inputs");
 *
 * Example Error Output (Debug build):
 *   [TENSOR VALIDATION ERROR] after_q_proj: Shape mismatch
 *     Expected: Q[8, 896] [8, 896]
 *     Actual:   [896, 8]
 *     Hint: Check for accidental transpose or incorrect projection dimensions
 *
 * @author David Sanftenberg
 */

#pragma once

#include "../tensors/Tensors.h"
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace llaminar2
{

    /**
     * @brief Tensor dimension specification
     *
     * Describes expected tensor shape at a specific pipeline stage.
     * Pipelines create TensorSpec instances with their architecture-specific dimensions.
     */
    struct TensorSpec
    {
        std::vector<size_t> expected_shape;
        std::string description;

        TensorSpec() = default;

        /**
         * @brief Create tensor specification
         *
         * @param shape Expected shape dimensions
         * @param desc Human-readable description (e.g., "Q[seq_len, n_heads*head_dim]")
         */
        TensorSpec(std::vector<size_t> shape, const std::string &desc = "")
            : expected_shape(std::move(shape)), description(desc) {}

        /**
         * @brief Check if tensor matches this specification (exact match)
         */
        bool matches(const std::vector<size_t> &actual_shape) const
        {
            if (actual_shape.size() != expected_shape.size())
                return false;

            for (size_t i = 0; i < actual_shape.size(); ++i)
            {
                if (actual_shape[i] != expected_shape[i])
                    return false;
            }
            return true;
        }

        /**
         * @brief Check if tensor can hold the expected data (relaxed match)
         *
         * Allows first dimension (typically seq_len) to be >= expected.
         * This is used for pre-allocated buffers sized for max_seq_len
         * but used with actual seq_len <= max_seq_len.
         *
         * Example: Buffer allocated as [2048, 896] can be validated against
         *          spec for [1, 896] since we only use first row.
         */
        bool matches_prefix(const std::vector<size_t> &actual_shape) const
        {
            if (actual_shape.size() != expected_shape.size())
                return false;

            for (size_t i = 0; i < actual_shape.size(); ++i)
            {
                if (i == 0)
                {
                    // First dimension: allow actual >= expected (buffer can be larger)
                    if (actual_shape[i] < expected_shape[i])
                        return false;
                }
                else
                {
                    // Other dimensions: require exact match
                    if (actual_shape[i] != expected_shape[i])
                        return false;
                }
            }
            return true;
        }

        /**
         * @brief Get human-readable shape string
         */
        std::string shape_str() const
        {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < expected_shape.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << expected_shape[i];
            }
            oss << "]";
            return oss.str();
        }

        /**
         * @brief Get shape string for actual tensor
         */
        static std::string shape_str(const std::vector<size_t> &shape)
        {
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i > 0)
                    oss << ", ";
                oss << shape[i];
            }
            oss << "]";
            return oss.str();
        }
    };

    // =============================================================================
    // Validation Macros (compile away in Release builds)
    // =============================================================================

#ifdef NDEBUG
// Release build: all validation compiles to nothing
#define VALIDATE_TENSOR(tensor, spec, stage) ((void)0)
#define VALIDATE_TENSOR_PTR(tensor_ptr, spec, stage) ((void)0)
#define VALIDATE_TENSOR_BUFFER(tensor, spec, stage) ((void)0)
#define VALIDATE_SHAPE(actual_shape, spec, stage) ((void)0)
#define ASSERT_SAME_SHAPE(tensor1, tensor2, stage) ((void)0)

#else
// Debug build: full validation with helpful error messages

/**
 * @brief Validate tensor shape matches specification
 *
 * @param tensor Shared pointer to TensorBase
 * @param spec TensorSpec describing expected shape
 * @param stage Pipeline stage name (for error messages)
 */
#define VALIDATE_TENSOR(tensor, spec, stage)                                                            \
    do                                                                                                  \
    {                                                                                                   \
        if (!(tensor))                                                                                  \
        {                                                                                               \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Tensor is null\n";               \
            std::cerr << "  Expected: " << (spec).description << " " << (spec).shape_str() << "\n";     \
            std::abort();                                                                               \
        }                                                                                               \
        const auto &_actual_shape = (tensor)->shape();                                                  \
        if (!(spec).matches(_actual_shape))                                                             \
        {                                                                                               \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Shape mismatch\n";               \
            std::cerr << "  Expected: " << (spec).description << " " << (spec).shape_str() << "\n";     \
            std::cerr << "  Actual:   " << TensorSpec::shape_str(_actual_shape) << "\n";                \
            std::cerr << "  Hint: Check for accidental transpose or incorrect projection dimensions\n"; \
            std::abort();                                                                               \
        }                                                                                               \
    } while (0)

/**
 * @brief Validate raw pointer tensor shape
 *
 * @param tensor_ptr Raw TensorBase pointer
 * @param spec TensorSpec describing expected shape
 * @param stage Pipeline stage name
 */
#define VALIDATE_TENSOR_PTR(tensor_ptr, spec, stage)                                                    \
    do                                                                                                  \
    {                                                                                                   \
        if (!(tensor_ptr))                                                                              \
        {                                                                                               \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Tensor pointer is null\n";       \
            std::cerr << "  Expected: " << (spec).description << " " << (spec).shape_str() << "\n";     \
            std::abort();                                                                               \
        }                                                                                               \
        const auto &_actual_shape = (tensor_ptr)->shape();                                              \
        if (!(spec).matches(_actual_shape))                                                             \
        {                                                                                               \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Shape mismatch\n";               \
            std::cerr << "  Expected: " << (spec).description << " " << (spec).shape_str() << "\n";     \
            std::cerr << "  Actual:   " << TensorSpec::shape_str(_actual_shape) << "\n";                \
            std::cerr << "  Hint: Check for accidental transpose or incorrect projection dimensions\n"; \
            std::abort();                                                                               \
        }                                                                                               \
    } while (0)

/**
 * @brief Validate pre-allocated buffer (allows first dimension >= expected)
 *
 * Use for buffers allocated with max_seq_len but used with actual seq_len.
 * First dimension can be larger (buffer capacity), other dimensions must match exactly.
 *
 * @param tensor Shared pointer to TensorBase (buffer)
 * @param spec TensorSpec describing minimum required shape
 * @param stage Pipeline stage name
 */
#define VALIDATE_TENSOR_BUFFER(tensor, spec, stage)                                                 \
    do                                                                                              \
    {                                                                                               \
        if (!(tensor))                                                                              \
        {                                                                                           \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Tensor is null\n";           \
            std::cerr << "  Expected: " << (spec).description << " " << (spec).shape_str() << "\n"; \
            std::abort();                                                                           \
        }                                                                                           \
        const auto &_actual_shape = (tensor)->shape();                                              \
        if (!(spec).matches_prefix(_actual_shape))                                                  \
        {                                                                                           \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Buffer too small\n";         \
            std::cerr << "  Required: " << (spec).description << " " << (spec).shape_str() << "\n"; \
            std::cerr << "  Actual:   " << TensorSpec::shape_str(_actual_shape) << "\n";            \
            std::cerr << "  Hint: Buffer first dimension must be >= required seq_len\n";            \
            std::abort();                                                                           \
        }                                                                                           \
    } while (0)

/**
 * @brief Validate raw shape vector
 *
 * @param actual_shape std::vector<size_t> actual shape
 * @param spec TensorSpec describing expected shape
 * @param stage Pipeline stage name
 */
#define VALIDATE_SHAPE(actual_shape, spec, stage)                                                       \
    do                                                                                                  \
    {                                                                                                   \
        if (!(spec).matches(actual_shape))                                                              \
        {                                                                                               \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Shape mismatch\n";               \
            std::cerr << "  Expected: " << (spec).description << " " << (spec).shape_str() << "\n";     \
            std::cerr << "  Actual:   " << TensorSpec::shape_str(actual_shape) << "\n";                 \
            std::cerr << "  Hint: Check for accidental transpose or incorrect projection dimensions\n"; \
            std::abort();                                                                               \
        }                                                                                               \
    } while (0)

/**
 * @brief Assert two tensors have the same shape
 *
 * @param tensor1 First tensor
 * @param tensor2 Second tensor
 * @param stage Pipeline stage name
 */
#define ASSERT_SAME_SHAPE(tensor1, tensor2, stage)                                                      \
    do                                                                                                  \
    {                                                                                                   \
        if (!(tensor1) || !(tensor2))                                                                   \
        {                                                                                               \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": One or both tensors are null\n"; \
            std::abort();                                                                               \
        }                                                                                               \
        const auto &_shape1 = (tensor1)->shape();                                                       \
        const auto &_shape2 = (tensor2)->shape();                                                       \
        if (_shape1.size() != _shape2.size())                                                           \
        {                                                                                               \
            std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Shape rank mismatch\n";          \
            std::cerr << "  Tensor1: " << TensorSpec::shape_str(_shape1) << "\n";                       \
            std::cerr << "  Tensor2: " << TensorSpec::shape_str(_shape2) << "\n";                       \
            std::abort();                                                                               \
        }                                                                                               \
        for (size_t _i = 0; _i < _shape1.size(); ++_i)                                                  \
        {                                                                                               \
            if (_shape1[_i] != _shape2[_i])                                                             \
            {                                                                                           \
                std::cerr << "[TENSOR VALIDATION ERROR] " << (stage) << ": Shape dimension mismatch\n"; \
                std::cerr << "  Tensor1: " << TensorSpec::shape_str(_shape1) << "\n";                   \
                std::cerr << "  Tensor2: " << TensorSpec::shape_str(_shape2) << "\n";                   \
                std::cerr << "  Hint: Tensors must have identical shapes for this operation\n";         \
                std::abort();                                                                           \
            }                                                                                           \
        }                                                                                               \
    } while (0)

#endif // NDEBUG

} // namespace llaminar2
