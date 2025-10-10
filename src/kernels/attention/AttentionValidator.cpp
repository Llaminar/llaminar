/**
 * @file AttentionValidator.cpp
 * @brief Implementation of attention validation utilities
 *
 * @author David Sanftenberg
 * @date October 7, 2025
 */

#include "AttentionValidator.h"
#include <cmath>
#include <algorithm>

namespace llaminar
{
    namespace attention
    {

        void AttentionValidator::scalarMatMul(const float *A, const float *B, float *C,
                                              size_t M, size_t N, size_t K,
                                              bool transpose_B)
        {
            // Initialize output to zero
            for (size_t i = 0; i < M; ++i)
            {
                float *c_row = C + i * N;
                for (size_t j = 0; j < N; ++j)
                {
                    c_row[j] = 0.0f;
                }
            }

            if (!transpose_B)
            {
                // Standard matmul: C[M,N] = A[M,K] @ B[K,N]
                // B is [K, N]: B[k,j] accessed as B[k*N + j]
                for (size_t i = 0; i < M; ++i)
                {
                    const float *a_row = A + i * K;
                    float *c_row = C + i * N;
                    for (size_t k = 0; k < K; ++k)
                    {
                        float a_val = a_row[k];
                        const float *b_row = B + k * N;
                        for (size_t j = 0; j < N; ++j)
                        {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
            else
            {
                // Transposed matmul: C[M,N] = A[M,K] @ B^T where B is [N,K]
                // B[j,k] accessed as B[j*K + k]
                for (size_t i = 0; i < M; ++i)
                {
                    const float *a_row = A + i * K;
                    float *c_row = C + i * N;
                    for (size_t j = 0; j < N; ++j)
                    {
                        const float *b_row = B + j * K;
                        float dot = 0.0f;
                        for (size_t k = 0; k < K; ++k)
                        {
                            dot += a_row[k] * b_row[k];
                        }
                        c_row[j] = dot;
                    }
                }
            }
        }

        ValidationResult AttentionValidator::validateProjection(
            const float *input, const float *weight, const float *computed,
            size_t M, size_t N, size_t K,
            bool transpose_B)
        {

            // Compute scalar reference
            std::vector<float> reference(M * N);
            scalarMatMul(input, weight, reference.data(), M, N, K, transpose_B);

            // Compare with computed result
            return compareTensors(reference.data(), computed, M * N);
        }

        ValidationResult AttentionValidator::compareTensors(
            const float *reference, const float *computed,
            size_t num_elements)
        {

            ValidationResult result;
            result.total_elements = num_elements;
            result.max_abs = 0.0;

            double sum_sq_ref = 0.0;
            double sum_sq_diff = 0.0;

            for (size_t i = 0; i < num_elements; ++i)
            {
                double ref_val = reference[i];
                double comp_val = computed[i];
                double diff = std::fabs(ref_val - comp_val);

                // Update max absolute error
                if (diff > result.max_abs)
                {
                    result.max_abs = diff;
                }

                // Accumulate for L2 norm
                sum_sq_ref += ref_val * ref_val;
                sum_sq_diff += diff * diff;
            }

            // Compute relative L2 norm
            result.rel_l2 = (sum_sq_ref == 0.0) ? 0.0 : std::sqrt(sum_sq_diff / sum_sq_ref);

            // Default tolerance check
            result.passed = isWithinTolerance(result);

            return result;
        }

        bool AttentionValidator::isWithinTolerance(const ValidationResult &result,
                                                   double max_abs_tol,
                                                   double rel_l2_tol)
        {
            return (result.max_abs <= max_abs_tol) && (result.rel_l2 <= rel_l2_tol);
        }

    } // namespace attention
} // namespace llaminar
