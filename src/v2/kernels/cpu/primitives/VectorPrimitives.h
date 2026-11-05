/**
 * @file VectorPrimitives.h
 * @brief ISA-dispatched vector primitives: dot, axpy, scale
 *
 * Native AArch64 NEON and runtime-autoselected x86 scalar / AVX2 / AVX-512
 * implementations for FP32 operations used by MoE routing, scatter/combine,
 * and gating stages. Buffers may be unaligned; n is nonnegative. Elementwise
 * operations permit exact input/output aliasing, but not partial overlap.
 * A zero-length call does not access input or elementwise output buffers.
 */

#pragma once

#include <cstddef>

namespace llaminar2::primitives
{
    /**
     * @brief Compute a dot product with an ISA-fixed reduction tree.
     * @param a First n-element input vector.
     * @param b Second n-element input vector.
     * @param n Number of live input elements; zero returns positive zero.
     * @return Sum of products in the selected ISA's arithmetic order.
     */
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

    /**
     * @brief Accumulate y[i] += alpha * x[i], with native FMA on SIMD paths.
     * @param y n-element in/out vector; may exactly alias x.
     * @param x n-element input vector.
     * @param alpha Multiplier applied to x.
     * @param n Number of live elements.
     */
    void vec_axpy(float *y, const float *x, float alpha, int n);

    /**
     * @brief Scale data[i] *= s in place.
     * @param data n-element in/out vector.
     * @param s Scaling factor.
     * @param n Number of live elements.
     */
    void vec_scale(float *data, float s, int n);

    /**
     * @brief Write out[i] = a[i] + b[i].
     * @param out n-element destination; may exactly alias either input.
     * @param a First n-element input vector.
     * @param b Second n-element input vector.
     * @param n Number of live elements.
     */
    void vec_add(float *out, const float *a, const float *b, int n);

} // namespace llaminar2::primitives
