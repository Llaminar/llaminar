/**
 * @file VectorPrimitives.cpp
 * @brief ISA-dispatched vector primitives: dot, axpy, scale
 *
 * AArch64 executes native NEON; x86 retains runtime scalar / AVX2 / AVX-512
 * dispatch via CPUFeatures.h. Grouped dot products share weight loads but
 * retain the serial operation's independent arithmetic chains and final fold,
 * so routing cannot change with batch size. ARM tails use explicit FMA to
 * prevent compiler contraction choices from changing serial/grouped bytes.
 */

#include "VectorPrimitives.h"
#include "../../../utils/CPUFeatures.h"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#include <array>
#include <cmath>
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives
{
#if defined(__aarch64__) && defined(__ARM_NEON)
    /**
     * @brief Share NEON weight loads across a compile-time number of dot products.
     *
     * Each row owns two four-lane FMA chains regardless of batch size. The
     * identical pairwise horizontal fold and explicit scalar FMA tail give
     * grouped execution the serial call's exact arithmetic, without row replay.
     * Eight live accumulators for four rows fit the AArch64 SIMD register file.
     * @tparam Rows One serial row or four grouped rows.
     * @param weights Shared n-element weight vector.
     * @param rows First activation row, with row_stride floats between rows.
     * @param n Number of live elements in every row.
     * @param row_stride Activation row spacing in floats.
     * @param output First destination scalar, with output_stride spacing.
     * @param output_stride Destination spacing in floats.
     */
    template <std::size_t Rows>
    static void vec_dot_neon_rows(
        const float *weights, const float *rows, int n,
        std::size_t row_stride, float *output, std::size_t output_stride)
    {
        static_assert(Rows == 1 || Rows == 4);
        std::array<float32x4_t, Rows> low{}, high{};
        int i = 0;
        for (; i < (n & ~7); i += 8)
        {
            const float32x4_t w0 = vld1q_f32(weights + i);
            const float32x4_t w1 = vld1q_f32(weights + i + 4);
            for (std::size_t row = 0; row < Rows; ++row)
            {
                const float *input = rows + row * row_stride + i;
                low[row] = vfmaq_f32(low[row], w0, vld1q_f32(input));
                high[row] = vfmaq_f32(high[row], w1, vld1q_f32(input + 4));
            }
        }
        for (std::size_t row = 0; row < Rows; ++row)
        {
            float value = vaddvq_f32(vaddq_f32(low[row], high[row]));
            for (int tail = i; tail < n; ++tail)
                value = std::fma(weights[tail], rows[row * row_stride + tail], value);
            output[row * output_stride] = value;
        }
    }

    float vec_dot(const float *a, const float *b, int n)
    {
        float value;
        vec_dot_neon_rows<1>(a, b, n, 0, &value, 0);
        return value;
    }

    void vec_dot_four_rows(
        const float *weights, const float *rows, int n,
        std::size_t row_stride, float *output, std::size_t output_stride)
    {
        vec_dot_neon_rows<4>(weights, rows, n, row_stride, output, output_stride);
    }

    void vec_axpy(float *y, const float *x, float alpha, int n)
    {
        const float32x4_t multiplier = vdupq_n_f32(alpha);
        int i = 0;
        for (; i < (n & ~7); i += 8)
        {
            const float32x4_t low = vfmaq_f32(vld1q_f32(y + i), multiplier, vld1q_f32(x + i));
            const float32x4_t high = vfmaq_f32(vld1q_f32(y + i + 4), multiplier, vld1q_f32(x + i + 4));
            vst1q_f32(y + i, low);
            vst1q_f32(y + i + 4, high);
        }
        for (; i < n; ++i)
            y[i] = std::fma(alpha, x[i], y[i]);
    }

    void vec_scale(float *data, float s, int n)
    {
        const float32x4_t multiplier = vdupq_n_f32(s);
        int i = 0;
        for (; i < (n & ~7); i += 8)
        {
            vst1q_f32(data + i, vmulq_f32(vld1q_f32(data + i), multiplier));
            vst1q_f32(data + i + 4, vmulq_f32(vld1q_f32(data + i + 4), multiplier));
        }
        for (; i < n; ++i)
            data[i] *= s;
    }

    void vec_add(float *out, const float *a, const float *b, int n)
    {
        int i = 0;
        for (; i < (n & ~7); i += 8)
        {
            const float32x4_t low = vaddq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
            const float32x4_t high = vaddq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
            vst1q_f32(out + i, low);
            vst1q_f32(out + i + 4, high);
        }
        for (; i < n; ++i)
            out[i] = a[i] + b[i];
    }
#else
    // ========================================================================
    // vec_dot — scalar / AVX2 / AVX-512
    // ========================================================================

    static float vec_dot_scalar(const float *a, const float *b, int n)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }

#if defined(__AVX2__)
    static float vec_dot_avx2(const float *a, const float *b, int n)
    {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        int i = 0;
        const int n16 = n & ~15;
        for (; i < n16; i += 16)
        {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        }
        acc0 = _mm256_add_ps(acc0, acc1);
        // Horizontal sum
        __m128 hi = _mm256_extractf128_ps(acc0, 1);
        __m128 lo = _mm256_castps256_ps128(acc0);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, lo);
        lo = _mm_add_ss(lo, shuf);
        float sum = _mm_cvtss_f32(lo);
        // Tail
        for (; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
#endif

#if defined(__AVX512F__)
    static float vec_dot_avx512(const float *a, const float *b, int n)
    {
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        int i = 0;
        const int n32 = n & ~31;
        for (; i < n32; i += 32)
        {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
        }
        acc0 = _mm512_add_ps(acc0, acc1);
        float sum = _mm512_reduce_add_ps(acc0);
        // Tail
        for (; i < n; ++i)
            sum += a[i] * b[i];
        return sum;
    }
#endif

    // Fallback stubs
#if !defined(__AVX2__)
    static float vec_dot_avx2(const float *a, const float *b, int n)
    {
        return vec_dot_scalar(a, b, n);
    }
#endif
#if !defined(__AVX512F__)
    static float vec_dot_avx512(const float *a, const float *b, int n)
    {
        return vec_dot_avx2(a, b, n);
    }
#endif

    float vec_dot(const float *a, const float *b, int n)
    {
        return ISA_DISPATCH_RETVAL(vec_dot, a, b, n);
    }

    /** @brief Scalar four-row implementation with unchanged serial arithmetic. */
    static void vec_dot_four_rows_scalar(
        const float *weights, const float *rows, int n,
        std::size_t row_stride, float *output, std::size_t output_stride)
    {
        for (int row = 0; row < 4; ++row)
            output[row * output_stride] =
                vec_dot_scalar(weights, rows + row * row_stride, n);
    }

#if defined(__AVX2__)
    /** @brief Fold the two eight-lane chains exactly as serial AVX2 vec_dot. */
    static inline float fold_dot_avx2(__m256 low, __m256 high)
    {
        const __m256 sum = _mm256_add_ps(low, high);
        __m128 folded = _mm_add_ps(
            _mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
        folded = _mm_add_ps(folded, _mm_movehdup_ps(folded));
        return _mm_cvtss_f32(
            _mm_add_ss(folded, _mm_movehl_ps(folded, folded)));
    }

    /**
     * @brief Reuse each AVX2 weight load across four serial-equivalent rows.
     * Eight independent chains hide FMA dependency latency while fitting the
     * sixteen-register ISA. Do not reassociate either chain or its final fold.
     */
    static void vec_dot_four_rows_avx2(
        const float *weights, const float *rows, int n,
        std::size_t row_stride, float *output, std::size_t output_stride)
    {
        __m256 a0 = _mm256_setzero_ps(), b0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps(), b1 = _mm256_setzero_ps();
        __m256 a2 = _mm256_setzero_ps(), b2 = _mm256_setzero_ps();
        __m256 a3 = _mm256_setzero_ps(), b3 = _mm256_setzero_ps();
        const float *r1 = rows + row_stride;
        const float *r2 = r1 + row_stride;
        const float *r3 = r2 + row_stride;
        int i = 0;
        for (; i < (n & ~15); i += 16)
        {
            const __m256 w0 = _mm256_loadu_ps(weights + i);
            const __m256 w1 = _mm256_loadu_ps(weights + i + 8);
            a0 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(rows + i), a0);
            b0 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(rows + i + 8), b0);
            a1 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(r1 + i), a1);
            b1 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(r1 + i + 8), b1);
            a2 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(r2 + i), a2);
            b2 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(r2 + i + 8), b2);
            a3 = _mm256_fmadd_ps(w0, _mm256_loadu_ps(r3 + i), a3);
            b3 = _mm256_fmadd_ps(w1, _mm256_loadu_ps(r3 + i + 8), b3);
        }
        float values[4] = {fold_dot_avx2(a0, b0), fold_dot_avx2(a1, b1),
                           fold_dot_avx2(a2, b2), fold_dot_avx2(a3, b3)};
        // Keep the scalar tail after the same final SIMD fold as M=1.
        for (int row = 0; row < 4; ++row)
        {
            for (int tail = i; tail < n; ++tail)
                values[row] += weights[tail] * rows[row * row_stride + tail];
            output[row * output_stride] = values[row];
        }
    }
#endif

#if defined(__AVX512F__)
    /** @brief AVX512 four-row dot with the serial two-chain reduction tree. */
    static void vec_dot_four_rows_avx512(
        const float *weights, const float *rows, int n,
        std::size_t row_stride, float *output, std::size_t output_stride)
    {
        __m512 a0 = _mm512_setzero_ps(), b0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), b2 = _mm512_setzero_ps();
        __m512 a3 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
        const float *r1 = rows + row_stride;
        const float *r2 = r1 + row_stride;
        const float *r3 = r2 + row_stride;
        int i = 0;
        for (; i < (n & ~31); i += 32)
        {
            const __m512 w0 = _mm512_loadu_ps(weights + i);
            const __m512 w1 = _mm512_loadu_ps(weights + i + 16);
            a0 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(rows + i), a0);
            b0 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(rows + i + 16), b0);
            a1 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(r1 + i), a1);
            b1 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(r1 + i + 16), b1);
            a2 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(r2 + i), a2);
            b2 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(r2 + i + 16), b2);
            a3 = _mm512_fmadd_ps(w0, _mm512_loadu_ps(r3 + i), a3);
            b3 = _mm512_fmadd_ps(w1, _mm512_loadu_ps(r3 + i + 16), b3);
        }
        float values[4] = {
            _mm512_reduce_add_ps(_mm512_add_ps(a0, b0)),
            _mm512_reduce_add_ps(_mm512_add_ps(a1, b1)),
            _mm512_reduce_add_ps(_mm512_add_ps(a2, b2)),
            _mm512_reduce_add_ps(_mm512_add_ps(a3, b3))};
        for (int row = 0; row < 4; ++row)
        {
            for (int tail = i; tail < n; ++tail)
                values[row] += weights[tail] * rows[row * row_stride + tail];
            output[row * output_stride] = values[row];
        }
    }
#endif

    void vec_dot_four_rows(
        const float *weights, const float *rows, int n,
        std::size_t row_stride, float *output, std::size_t output_stride)
    {
        ISA_DISPATCH_VOID(vec_dot_four_rows, weights, rows, n,
                          row_stride, output, output_stride);
    }

    // ========================================================================
    // vec_axpy — y += alpha * x
    // ========================================================================

    static void vec_axpy_scalar(float *y, const float *x, float alpha, int n)
    {
        for (int i = 0; i < n; ++i)
            y[i] += alpha * x[i];
    }

#if defined(__AVX2__)
    static void vec_axpy_avx2(float *y, const float *x, float alpha, int n)
    {
        const __m256 va = _mm256_set1_ps(alpha);
        int i = 0;
        const int n16 = n & ~15;
        for (; i < n16; i += 16)
        {
            __m256 y0 = _mm256_loadu_ps(y + i);
            __m256 y1 = _mm256_loadu_ps(y + i + 8);
            y0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), y0);
            y1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i + 8), y1);
            _mm256_storeu_ps(y + i, y0);
            _mm256_storeu_ps(y + i + 8, y1);
        }
        for (; i < n; ++i)
            y[i] += alpha * x[i];
    }
#endif

#if defined(__AVX512F__)
    static void vec_axpy_avx512(float *y, const float *x, float alpha, int n)
    {
        const __m512 va = _mm512_set1_ps(alpha);
        int i = 0;
        const int n32 = n & ~31;
        for (; i < n32; i += 32)
        {
            __m512 y0 = _mm512_loadu_ps(y + i);
            __m512 y1 = _mm512_loadu_ps(y + i + 16);
            y0 = _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i), y0);
            y1 = _mm512_fmadd_ps(va, _mm512_loadu_ps(x + i + 16), y1);
            _mm512_storeu_ps(y + i, y0);
            _mm512_storeu_ps(y + i + 16, y1);
        }
        for (; i < n; ++i)
            y[i] += alpha * x[i];
    }
#endif

#if !defined(__AVX2__)
    static void vec_axpy_avx2(float *y, const float *x, float alpha, int n)
    {
        vec_axpy_scalar(y, x, alpha, n);
    }
#endif
#if !defined(__AVX512F__)
    static void vec_axpy_avx512(float *y, const float *x, float alpha, int n)
    {
        vec_axpy_avx2(y, x, alpha, n);
    }
#endif

    void vec_axpy(float *y, const float *x, float alpha, int n)
    {
        ISA_DISPATCH_VOID(vec_axpy, y, x, alpha, n);
    }

    // ========================================================================
    // vec_scale — data *= s
    // ========================================================================

    static void vec_scale_scalar(float *data, float s, int n)
    {
        for (int i = 0; i < n; ++i)
            data[i] *= s;
    }

#if defined(__AVX2__)
    static void vec_scale_avx2(float *data, float s, int n)
    {
        const __m256 vs = _mm256_set1_ps(s);
        int i = 0;
        const int n16 = n & ~15;
        for (; i < n16; i += 16)
        {
            _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vs));
            _mm256_storeu_ps(data + i + 8, _mm256_mul_ps(_mm256_loadu_ps(data + i + 8), vs));
        }
        for (; i < n; ++i)
            data[i] *= s;
    }
#endif

#if defined(__AVX512F__)
    static void vec_scale_avx512(float *data, float s, int n)
    {
        const __m512 vs = _mm512_set1_ps(s);
        int i = 0;
        const int n32 = n & ~31;
        for (; i < n32; i += 32)
        {
            _mm512_storeu_ps(data + i, _mm512_mul_ps(_mm512_loadu_ps(data + i), vs));
            _mm512_storeu_ps(data + i + 16, _mm512_mul_ps(_mm512_loadu_ps(data + i + 16), vs));
        }
        for (; i < n; ++i)
            data[i] *= s;
    }
#endif

#if !defined(__AVX2__)
    static void vec_scale_avx2(float *data, float s, int n)
    {
        vec_scale_scalar(data, s, n);
    }
#endif
#if !defined(__AVX512F__)
    static void vec_scale_avx512(float *data, float s, int n)
    {
        vec_scale_avx2(data, s, n);
    }
#endif

    void vec_scale(float *data, float s, int n)
    {
        ISA_DISPATCH_VOID(vec_scale, data, s, n);
    }

    // ========================================================================
    // vec_add — out = a + b
    // ========================================================================

    static void vec_add_scalar(float *out, const float *a, const float *b, int n)
    {
        for (int i = 0; i < n; ++i)
            out[i] = a[i] + b[i];
    }

#if defined(__AVX2__)
    static void vec_add_avx2(float *out, const float *a, const float *b, int n)
    {
        int i = 0;
        const int n16 = n & ~15;
        for (; i < n16; i += 16)
        {
            _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
            _mm256_storeu_ps(out + i + 8, _mm256_add_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8)));
        }
        for (; i < n; ++i)
            out[i] = a[i] + b[i];
    }
#endif

#if defined(__AVX512F__)
    static void vec_add_avx512(float *out, const float *a, const float *b, int n)
    {
        int i = 0;
        const int n32 = n & ~31;
        for (; i < n32; i += 32)
        {
            _mm512_storeu_ps(out + i, _mm512_add_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i)));
            _mm512_storeu_ps(out + i + 16, _mm512_add_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16)));
        }
        for (; i < n; ++i)
            out[i] = a[i] + b[i];
    }
#endif

#if !defined(__AVX2__)
    static void vec_add_avx2(float *out, const float *a, const float *b, int n)
    {
        vec_add_scalar(out, a, b, n);
    }
#endif
#if !defined(__AVX512F__)
    static void vec_add_avx512(float *out, const float *a, const float *b, int n)
    {
        vec_add_avx2(out, a, b, n);
    }
#endif

    void vec_add(float *out, const float *a, const float *b, int n)
    {
        ISA_DISPATCH_VOID(vec_add, out, a, b, n);
    }

#endif // AArch64 NEON / x86 runtime dispatch
} // namespace llaminar2::primitives
