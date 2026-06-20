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

    /// @brief AXPY: y[i] += alpha * x[i] for i in [0, n)
    void vec_axpy(float *y, const float *x, float alpha, int n);

    /// @brief Scale in-place: data[i] *= s for i in [0, n)
    void vec_scale(float *data, float s, int n);

    /// @brief Vector add: out[i] = a[i] + b[i] for i in [0, n)
    void vec_add(float *out, const float *a, const float *b, int n);

} // namespace llaminar2::primitives
