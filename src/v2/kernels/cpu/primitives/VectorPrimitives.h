/**
 * @file VectorPrimitives.h
 * @brief ISA-dispatched vector primitives: dot, axpy, scale
 *
 * Runtime-autoselected scalar / AVX2 / AVX-512 implementations for
 * small FP32 vector operations used by MoE routing, scatter/combine,
 * and gating stages.
 */

#pragma once

#include <cstddef>

namespace llaminar2::primitives
{
    /// @brief Dot product: returns sum(a[i] * b[i]) for i in [0, n)
    float vec_dot(const float *a, const float *b, int n);

    /**
     * @brief Four independent dot products sharing one immutable weight row.
     *
     * Reuses each weight load across four activation rows, without changing
     * the active ISA's serial vec_dot accumulator chains, fold, or tail order.
     * Inputs and output must not overlap; the caller owns valid full rows.
     * No allocation or worker-team creation occurs in this primitive.
     * @param weights Shared n-element weight row.
     * @param rows First of four activation rows.
     * @param n Number of elements per dot product.
     * @param row_stride Distance between activation rows, in floats.
     * @param output First destination scalar.
     * @param output_stride Distance between destination scalars, in floats.
     */
    void vec_dot_four_rows(const float *weights, const float *rows, int n,
                           std::size_t row_stride, float *output,
                           std::size_t output_stride);

    /// @brief AXPY: y[i] += alpha * x[i] for i in [0, n)
    void vec_axpy(float *y, const float *x, float alpha, int n);

    /// @brief Scale in-place: data[i] *= s for i in [0, n)
    void vec_scale(float *data, float s, int n);

    /// @brief Vector add: out[i] = a[i] + b[i] for i in [0, n)
    void vec_add(float *out, const float *a, const float *b, int n);

} // namespace llaminar2::primitives
