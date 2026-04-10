/**
 * @file SoftmaxPrimitives.cpp
 * @brief Vectorized Softmax implementation (ported from V1)
 * @author David Sanftenberg
 */

#include "SoftmaxPrimitives.h"
#include <algorithm>
#include <cmath>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../../utils/OpenMPUtils.h"
#include "../../../utils/CPUFeatures.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives
{

    namespace detail
    {

#if defined(__AVX2__)
        // Horizontal max reduction for AVX2
        static inline float hmax256(__m256 v)
        {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 m = _mm_max_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(m);
            m = _mm_max_ps(m, shuf);
            shuf = _mm_movehl_ps(shuf, m);
            m = _mm_max_ps(m, shuf);
            return _mm_cvtss_f32(m);
        }
#endif

#if defined(__AVX512F__) || defined(__AVX2__)
        // Fast exp approximation (polynomial)
        static inline __m512 fast_exp_approx512(__m512 x)
        {
#if defined(__AVX512F__)
            const __m512 max_clip = _mm512_set1_ps(10.0f);
            const __m512 min_clip = _mm512_set1_ps(-20.0f);
            x = _mm512_max_ps(min_clip, _mm512_min_ps(max_clip, x));

            const __m512 inv_ln2 = _mm512_set1_ps(1.4426950408889634f);
            __m512 xf = _mm512_mul_ps(x, inv_ln2);
            __m512 fx = _mm512_floor_ps(xf);
            __m512 fpart = _mm512_sub_ps(xf, fx);

            const __m512 c1 = _mm512_set1_ps(0.99999994f);
            const __m512 c2 = _mm512_set1_ps(0.69314718f);
            const __m512 c3 = _mm512_set1_ps(0.24022651f);
            const __m512 c4 = _mm512_set1_ps(0.05550411f);
            const __m512 c5 = _mm512_set1_ps(0.00961813f);

            __m512 p = c5;
            p = _mm512_fmadd_ps(p, fpart, c4);
            p = _mm512_fmadd_ps(p, fpart, c3);
            p = _mm512_fmadd_ps(p, fpart, c2);
            p = _mm512_fmadd_ps(p, fpart, c1);

            __m512i ipart = _mm512_cvttps_epi32(fx);
            ipart = _mm512_add_epi32(ipart, _mm512_set1_epi32(127));
            ipart = _mm512_slli_epi32(ipart, 23);
            __m512 two_ip = _mm512_castsi512_ps(ipart);

            return _mm512_mul_ps(two_ip, p);
#else
            return x;
#endif
        }

        static inline __m256 fast_exp_approx256(__m256 x)
        {
#if defined(__AVX2__)
            const __m256 max_clip = _mm256_set1_ps(10.0f);
            const __m256 min_clip = _mm256_set1_ps(-20.0f);
            x = _mm256_max_ps(min_clip, _mm256_min_ps(max_clip, x));

            const __m256 inv_ln2 = _mm256_set1_ps(1.4426950408889634f);
            __m256 xf = _mm256_mul_ps(x, inv_ln2);
            __m256 fx = _mm256_floor_ps(xf);
            __m256 fpart = _mm256_sub_ps(xf, fx);

            const __m256 c1 = _mm256_set1_ps(0.99999994f);
            const __m256 c2 = _mm256_set1_ps(0.69314718f);
            const __m256 c3 = _mm256_set1_ps(0.24022651f);
            const __m256 c4 = _mm256_set1_ps(0.05550411f);
            const __m256 c5 = _mm256_set1_ps(0.00961813f);

            __m256 p = c5;
            p = _mm256_fmadd_ps(p, fpart, c4);
            p = _mm256_fmadd_ps(p, fpart, c3);
            p = _mm256_fmadd_ps(p, fpart, c2);
            p = _mm256_fmadd_ps(p, fpart, c1);

            __m256i ipart = _mm256_cvttps_epi32(fx);
            ipart = _mm256_add_epi32(ipart, _mm256_set1_epi32(127));
            ipart = _mm256_slli_epi32(ipart, 23);
            __m256 two_ip = _mm256_castsi256_ps(ipart);

            return _mm256_mul_ps(two_ip, p);
#else
            return x;
#endif
        }
#endif

        // ======== Per-row softmax helpers: scalar, AVX2, AVX512 ========

        void softmax_row_scalar(float *row, int cols, long long r,
                                bool causal, float scale, bool use_fast_exp)
        {
            float row_max = -std::numeric_limits<float>::infinity();
#pragma omp simd reduction(max : row_max)
            for (long long c = 0; c < (long long)cols; ++c)
            {
                bool masked = causal && c > r;
                float v = row[c];
                if (scale != 1.0f)
                    v *= scale;
                if (masked)
                    v = -std::numeric_limits<float>::infinity();
                row_max = v > row_max ? v : row_max;
            }

            if (!std::isfinite(row_max))
                row_max = 0.0f;

            double sum = 0.0;
#pragma omp simd reduction(+ : sum)
            for (long long c = 0; c < (long long)cols; ++c)
            {
                bool masked = causal && c > r;
                float v = row[c];
                if (scale != 1.0f)
                    v *= scale;
                if (masked)
                    continue;
                sum += std::exp(v - row_max);
            }

            if (sum <= 0.0)
                sum = 1.0;
            float inv = (float)(1.0 / sum);

#pragma omp simd
            for (long long c = 0; c < (long long)cols; ++c)
            {
                bool masked = causal && c > r;
                float v = row[c];
                if (scale != 1.0f)
                    v *= scale;
                if (masked)
                {
                    row[c] = 0.0f;
                    continue;
                }
                row[c] = (float)(std::exp(v - row_max) * inv);
            }
        }

#if defined(__AVX2__)
        void softmax_row_avx2(float *row, int cols, long long r,
                              bool causal, float scale, bool use_fast_exp)
        {
            float row_max = -std::numeric_limits<float>::infinity();

            // Pass 1: Find max
            int c = 0;
            __m256 vmax = _mm256_set1_ps(-std::numeric_limits<float>::infinity());

            for (; c + 8 <= cols; c += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c);
                if (scale != 1.0f)
                    v = _mm256_mul_ps(v, _mm256_set1_ps(scale));

                if (causal)
                {
                    float tmp[8];
                    _mm256_storeu_ps(tmp, v);
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        if (c + lane > r)
                            tmp[lane] = -std::numeric_limits<float>::infinity();
                    }
                    v = _mm256_loadu_ps(tmp);
                }

                vmax = _mm256_max_ps(vmax, v);
            }

            row_max = std::max(row_max, hmax256(vmax));

            for (; c < cols; ++c)
            {
                bool masked = causal && c > r;
                float v = row[c];
                if (scale != 1.0f)
                    v *= scale;
                if (masked)
                    v = -std::numeric_limits<float>::infinity();
                row_max = v > row_max ? v : row_max;
            }

            if (!std::isfinite(row_max))
                row_max = 0.0f;

            // Pass 2: Compute sum of exp(x - max)
            double sum = 0.0;
            int c2 = 0;
            for (; c2 + 8 <= cols; c2 += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c2);
                if (scale != 1.0f)
                    v = _mm256_mul_ps(v, _mm256_set1_ps(scale));

                float tmp[8];
                _mm256_storeu_ps(tmp, v);

                for (int lane = 0; lane < 8; ++lane)
                {
                    if (causal && c2 + lane > r)
                        continue;
                    sum += std::exp(tmp[lane] - row_max);
                }
            }

            for (; c2 < cols; ++c2)
            {
                if (causal && c2 > r)
                    continue;
                float v = row[c2];
                if (scale != 1.0f)
                    v *= scale;
                sum += std::exp(v - row_max);
            }

            if (sum <= 0.0)
                sum = 1.0;
            float inv = (float)(1.0 / sum);

            // Pass 3: Normalize
            int c3 = 0;
            for (; c3 + 8 <= cols; c3 += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c3);
                if (scale != 1.0f)
                    v = _mm256_mul_ps(v, _mm256_set1_ps(scale));

                float tmp[8];
                _mm256_storeu_ps(tmp, v);

                float out[8];
                for (int lane = 0; lane < 8; ++lane)
                {
                    if (causal && c3 + lane > r)
                    {
                        out[lane] = 0.0f;
                    }
                    else
                    {
                        float dv = tmp[lane] - row_max;
                        if (use_fast_exp)
                        {
                            __m256 x = _mm256_set1_ps(dv);
                            __m256 e = fast_exp_approx256(x);
                            _mm256_storeu_ps(tmp, e);
                            out[lane] = tmp[0] * inv;
                        }
                        else
                        {
                            out[lane] = std::exp(dv) * inv;
                        }
                    }
                }

                _mm256_storeu_ps(row + c3, _mm256_loadu_ps(out));
            }

            for (; c3 < cols; ++c3)
            {
                if (causal && c3 > r)
                {
                    row[c3] = 0.0f;
                    continue;
                }
                float v = row[c3];
                if (scale != 1.0f)
                    v *= scale;
                row[c3] = std::exp(v - row_max) * inv;
            }
        }
#endif

#if defined(__AVX512F__)
        void softmax_row_avx512(float *row, int cols, long long r,
                                bool causal, float scale, bool use_fast_exp)
        {
            float row_max = -std::numeric_limits<float>::infinity();

            // Pass 1: Find max
            int c = 0;
            const __m512 neg_inf = _mm512_set1_ps(-std::numeric_limits<float>::infinity());
            const __m512 scale_ps = _mm512_set1_ps(scale);
            __m512 vmax = neg_inf;

            for (; c + 16 <= cols; c += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c);
                if (scale != 1.0f)
                    v = _mm512_mul_ps(v, scale_ps);

                if (causal)
                {
                    for (int lane = 0; lane < 16; ++lane)
                    {
                        if (c + lane > r)
                        {
                            alignas(64) float tmp[16];
                            _mm512_store_ps(tmp, v);
                            tmp[lane] = -std::numeric_limits<float>::infinity();
                            v = _mm512_load_ps(tmp);
                        }
                    }
                }

                vmax = _mm512_max_ps(vmax, v);
            }

            row_max = std::max(row_max, _mm512_reduce_max_ps(vmax));

            for (; c < cols; ++c)
            {
                bool masked = causal && c > r;
                float v = row[c];
                if (scale != 1.0f)
                    v *= scale;
                if (masked)
                    v = -std::numeric_limits<float>::infinity();
                if (v > row_max)
                    row_max = v;
            }

            if (!std::isfinite(row_max))
                row_max = 0.0f;

            // Pass 2: Compute sum of exp(x - max)
            double sum = 0.0;
            int c2 = 0;
            for (; c2 + 16 <= cols; c2 += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c2);
                if (scale != 1.0f)
                    v = _mm512_mul_ps(v, scale_ps);

                int n_valid = causal ? ((int)r - c2 + 1) : 16;
                if (n_valid <= 0)
                    continue;

                alignas(64) float tmp[16];
                _mm512_store_ps(tmp, v);

                for (int lane = 0; lane < 16; ++lane)
                {
                    if (causal && c2 + lane > r)
                        continue;
                    sum += std::exp(tmp[lane] - row_max);
                }
            }

            for (; c2 < cols; ++c2)
            {
                if (causal && c2 > r)
                    continue;
                float v = row[c2];
                if (scale != 1.0f)
                    v *= scale;
                sum += std::exp(v - row_max);
            }

            if (sum <= 0.0)
                sum = 1.0;
            float inv = (float)(1.0 / sum);

            // Pass 3: Normalize
            int c3 = 0;
            const __m512 inv_ps = _mm512_set1_ps(inv);
            const __m512 zero_ps = _mm512_setzero_ps();

            for (; c3 + 16 <= cols; c3 += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c3);
                if (scale != 1.0f)
                    v = _mm512_mul_ps(v, scale_ps);

                int n_valid = causal ? ((int)r - c3 + 1) : 16;
                if (n_valid <= 0)
                {
                    _mm512_storeu_ps(row + c3, zero_ps);
                    continue;
                }

                alignas(64) float tmp[16];
                _mm512_store_ps(tmp, v);

                float out[16];
                for (int lane = 0; lane < 16; ++lane)
                {
                    if (causal && c3 + lane > r)
                    {
                        out[lane] = 0.0f;
                    }
                    else
                    {
                        float dv = tmp[lane] - row_max;
                        if (use_fast_exp)
                        {
                            __m512 x = _mm512_set1_ps(dv);
                            __m512 e = fast_exp_approx512(x);
                            _mm512_store_ps(tmp, e);
                            out[lane] = tmp[0] * inv;
                        }
                        else
                        {
                            out[lane] = std::exp(dv) * inv;
                        }
                    }
                }

                _mm512_storeu_ps(row + c3, _mm512_loadu_ps(out));
            }

            for (; c3 < cols; ++c3)
            {
                if (causal && c3 > r)
                {
                    row[c3] = 0.0f;
                    continue;
                }
                float v = row[c3];
                if (scale != 1.0f)
                    v *= scale;
                row[c3] = std::exp(v - row_max) * inv;
            }
        }
#endif

// --- ISA stubs for runtime dispatch ---
#if !defined(__AVX2__)
        void softmax_row_avx2(float *row, int cols, long long r,
                              bool causal, float scale, bool use_fast_exp)
        {
            softmax_row_scalar(row, cols, r, causal, scale, use_fast_exp);
        }
#endif
#if !defined(__AVX512F__)
        void softmax_row_avx512(float *row, int cols, long long r,
                                bool causal, float scale, bool use_fast_exp)
        {
            softmax_row_avx2(row, cols, r, causal, scale, use_fast_exp);
        }
#endif

    } // namespace detail
    using namespace detail;

    void softmax_row_major_vectorized(
        const SoftmaxRowArgs &args,
        const SoftmaxExecOptions &opts)
    {
        if (!args.scores || args.rows <= 0 || args.cols <= 0)
            return;

        float *scores = args.scores;
        const int rows = args.rows;
        const int cols = args.cols;
        const bool causal = args.causal;
        const float scale = args.scale;
        const bool use_fast_exp = opts.fast_exp;
        const bool force_scalar = opts.force_scalar;

        // Parallelization heuristic
        std::size_t total_elems = (std::size_t)rows * (std::size_t)cols;
        bool parallel = !force_scalar && rows > 1 && total_elems >= (std::size_t)opts.parallel_elems_threshold;
        if (opts.parallel_row_threshold > 0 && rows < opts.parallel_row_threshold)
            parallel = false;

        // Define the softmax work as a lambda for OMP_WORKSHARE_REGION
        auto do_softmax = [&]()
        {
#pragma omp for schedule(static)
            for (long long r = 0; r < (long long)rows; ++r)
            {
                float *row = scores + (std::size_t)r * cols;
                ISA_DISPATCH_VOID(softmax_row, row, cols, r, causal, scale, use_fast_exp);
            }
        }; // end do_softmax lambda

        // Execute with OpenMP worksharing (nested-safe)
        if (parallel)
        {
            OMP_WORKSHARE_REGION(do_softmax);
        }
        else
        {
            // Serial execution: run row loop without OpenMP
            for (long long r = 0; r < (long long)rows; ++r)
            {
                float *row = scores + (std::size_t)r * cols;
                ISA_DISPATCH_VOID(softmax_row, row, cols, r, causal, scale, use_fast_exp);
            }
        }
    }

} // namespace llaminar2::primitives
