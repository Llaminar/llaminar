/**
 * @file AttentionValidator.h
 * @brief Validation utilities for attention kernel projections and computations
 *
 * This class provides scalar reference implementations and validation logic
 * for verifying the correctness of optimized attention kernel operations.
 * Validation code is separated from production hot paths to enable:
 * - Zero overhead when validation is disabled
 * - Clean separation of concerns
 * - Reusable validation for intermediate tensors
 * - Systematic debugging of numerical divergences
 *
 * @author David Sanftenberg
 * @date October 7, 2025
 */

#pragma once

#include <cstddef>
#include <vector>
#include <memory>

namespace llaminar
{
    namespace attention
    {

        /**
         * @brief Validation result containing divergence metrics
         */
        struct ValidationResult
        {
            bool passed;           ///< Whether validation passed (within tolerance)
            double max_abs;        ///< Maximum absolute difference
            double rel_l2;         ///< Relative L2 norm of difference
            size_t total_elements; ///< Total elements compared
        };

        /**
         * @brief Utility class for validating attention kernel operations
         *
         * Provides scalar reference implementations for matrix multiplications
         * and validation utilities for comparing results against references.
         * All methods are static as this is a stateless utility class.
         *
         * Usage:
         * @code
         * // Validate a projection against scalar reference
         * auto result = AttentionValidator::validateProjection(
         *     input_ptr, weight_ptr, computed_output_ptr,
         *     seq_len, output_dim, input_dim,
         *     true  // transpose_B
         * );
         * if (!result.passed) {
         *     LOG_WARN("Divergence detected: max_abs=" << result.max_abs);
         * }
         * @endcode
         */
        class AttentionValidator
        {
        public:
            /**
             * @brief Compute scalar matrix multiplication reference
             *
             * Performs C = A @ B or C = A @ B^T depending on transpose_B flag.
             * This is a simple, correct implementation used as ground truth
             * for validating optimized implementations.
             *
             * @param A Input matrix A [M, K] (row-major)
             * @param B Input matrix B [K, N] or [N, K] if transpose_B (row-major)
             * @param C Output matrix C [M, N] (row-major, will be written)
             * @param M Number of rows in A and C
             * @param N Number of columns in B (or rows if transposed) and C
             * @param K Number of columns in A and rows in B (or columns if transposed)
             * @param transpose_B If true, B is [N, K] and we compute A @ B^T
             *
             * @note This is intentionally simple and unoptimized for correctness.
             *       Performance: O(M*N*K), suitable only for validation/debugging.
             */
            static void scalarMatMul(const float *A, const float *B, float *C,
                                     size_t M, size_t N, size_t K,
                                     bool transpose_B = false);

            /**
             * @brief Validate a projection operation against scalar reference
             *
             * Computes scalar reference and compares against provided result.
             * Returns validation metrics including max absolute error and relative L2 norm.
             *
             * @param input Input activations [M, K]
             * @param weight Weight matrix [N, K] (assuming transpose_B=true for PyTorch convention)
             * @param computed Computed output to validate [M, N]
             * @param M Number of rows (e.g., seq_len)
             * @param N Number of output features (e.g., head_dim)
             * @param K Number of input features (e.g., d_model)
             * @param transpose_B Whether weight is transposed (typically true for PyTorch nn.Linear)
             *
             * @return ValidationResult with divergence metrics
             *
             * @note Allocates M*N temporary buffer for reference computation
             */
            static ValidationResult validateProjection(
                const float *input, const float *weight, const float *computed,
                size_t M, size_t N, size_t K,
                bool transpose_B = true);

            /**
             * @brief Compare two tensors and compute divergence metrics
             *
             * Utility for comparing any two equal-sized tensors.
             * Computes max absolute error and relative L2 norm.
             *
             * @param reference Reference tensor (ground truth)
             * @param computed Computed tensor to validate
             * @param num_elements Number of elements in both tensors
             *
             * @return ValidationResult with divergence metrics
             */
            static ValidationResult compareTensors(
                const float *reference, const float *computed,
                size_t num_elements);

            /**
             * @brief Check if validation result passes given tolerances
             *
             * @param result Validation result to check
             * @param max_abs_tol Maximum absolute error tolerance (default: 1e-5)
             * @param rel_l2_tol Relative L2 norm tolerance (default: 1e-5)
             *
             * @return true if both max_abs and rel_l2 are within tolerance
             */
            static bool isWithinTolerance(const ValidationResult &result,
                                          double max_abs_tol = 1e-5,
                                          double rel_l2_tol = 1e-5);

        private:
            // Stateless utility class - no construction needed
            AttentionValidator() = delete;
            ~AttentionValidator() = delete;
            AttentionValidator(const AttentionValidator &) = delete;
            AttentionValidator &operator=(const AttentionValidator &) = delete;
        };

    } // namespace attention
} // namespace llaminar
