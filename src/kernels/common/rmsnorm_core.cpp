/**
 * @file rmsnorm_core.cpp
 * @brief Implementation of centralized RMSNorm math primitives.
 */
#include "rmsnorm_core.h"

#include <cmath>
#include <algorithm>
#include "../../utils/debug_env.h"
#include "../../logger.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar::kernels
{

    static inline bool want_parallel(std::size_t rows, std::size_t cols, const RMSNormExecOptions &opts)
    {
        if (opts.force_scalar || !opts.allow_parallel)
            return false;
        std::size_t elems = rows * cols;
        if (rows <= 1)
            return false; // decode / single-row path stays single-thread
#ifdef _OPENMP
        if (omp_in_parallel())
            return false;
#endif
        return elems >= opts.parallel_threshold_elems;
    }

    void rmsnorm_compute_row_sumsq(const float *src,
                                   std::size_t rows,
                                   std::size_t cols,
                                   double *row_sumsq,
                                   const RMSNormExecOptions &opts)
    {
        if (!src || !row_sumsq || rows == 0 || cols == 0)
            return;
        bool parallel = want_parallel(rows, cols, opts);
#if defined(__AVX2__) || defined(__AVX512F__)
        const auto &renv_top = llaminar::debugEnv().rmsnorm;
        bool fast_acc = renv_top.fast_accumulate;
#else
        bool fast_acc = false;
#endif
#pragma omp parallel for if (parallel)
        for (long long r = 0; r < (long long)rows; ++r)
        {
            const float *row = src + (std::size_t)r * cols;
            double acc = 0.0;
#if defined(__AVX512F__)
            const auto &renv = llaminar::debugEnv().rmsnorm;
            bool allow_avx512 = (renv.vec_impl == 0 || renv.vec_impl == 3);
            bool allow_avx2_fallback = (renv.vec_impl == 0 || renv.vec_impl == 2);
            if (allow_avx512)
            {
                // Multi-accumulator AVX512 path: process 64 elements per unrolled iteration
                long long c = 0;
                __m512d dacc0 = _mm512_setzero_pd();
                __m512d dacc1 = _mm512_setzero_pd();
                __m512d dacc2 = _mm512_setzero_pd();
                __m512d dacc3 = _mm512_setzero_pd();
                // Use float loads => widen to double for precision parity with original implementation.
                for (; c + 64 <= (long long)cols; c += 64)
                {
                    const float *base = row + c;
                    __m512 v0 = _mm512_loadu_ps(base + 0);
                    __m512 v1 = _mm512_loadu_ps(base + 16);
                    __m512 v2 = _mm512_loadu_ps(base + 32);
                    __m512 v3 = _mm512_loadu_ps(base + 48);
                    // Convert to double (split into low/high 8 lanes)
                    __m256 lo0 = _mm512_castps512_ps256(v0);
                    __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v0), 1));
                    __m256 lo1 = _mm512_castps512_ps256(v1);
                    __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v1), 1));
                    __m256 lo2 = _mm512_castps512_ps256(v2);
                    __m256 hi2 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v2), 1));
                    __m256 lo3 = _mm512_castps512_ps256(v3);
                    __m256 hi3 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v3), 1));
                    __m512d d0a = _mm512_cvtps_pd(lo0);
                    __m512d d0b = _mm512_cvtps_pd(hi0);
                    __m512d d1a = _mm512_cvtps_pd(lo1);
                    __m512d d1b = _mm512_cvtps_pd(hi1);
                    __m512d d2a = _mm512_cvtps_pd(lo2);
                    __m512d d2b = _mm512_cvtps_pd(hi2);
                    __m512d d3a = _mm512_cvtps_pd(lo3);
                    __m512d d3b = _mm512_cvtps_pd(hi3);
                    dacc0 = _mm512_fmadd_pd(d0a, d0a, dacc0);
                    dacc0 = _mm512_fmadd_pd(d0b, d0b, dacc0);
                    dacc1 = _mm512_fmadd_pd(d1a, d1a, dacc1);
                    dacc1 = _mm512_fmadd_pd(d1b, d1b, dacc1);
                    dacc2 = _mm512_fmadd_pd(d2a, d2a, dacc2);
                    dacc2 = _mm512_fmadd_pd(d2b, d2b, dacc2);
                    dacc3 = _mm512_fmadd_pd(d3a, d3a, dacc3);
                    dacc3 = _mm512_fmadd_pd(d3b, d3b, dacc3);
                }
                acc += _mm512_reduce_add_pd(dacc0) + _mm512_reduce_add_pd(dacc1) + _mm512_reduce_add_pd(dacc2) + _mm512_reduce_add_pd(dacc3);
                // Tail (process remaining with 16-float blocks or scalar)
                for (; c + 16 <= (long long)cols; c += 16)
                {
                    __m512 v = _mm512_loadu_ps(row + c);
                    __m256 vlo = _mm512_castps512_ps256(v);
                    __m256 vhi = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v), 1));
                    __m512d d0 = _mm512_cvtps_pd(vlo);
                    __m512d d1 = _mm512_cvtps_pd(vhi);
                    __m512d sq0 = _mm512_mul_pd(d0, d0);
                    __m512d sq1 = _mm512_mul_pd(d1, d1);
                    acc += _mm512_reduce_add_pd(sq0) + _mm512_reduce_add_pd(sq1);
                }
                if (!fast_acc)
                {
                    for (; c < (long long)cols; ++c)
                    {
                        double v = (double)row[c];
                        acc += v * v;
                    }
                }
                else
                {
                    float facc = 0.f;
                    for (; c < (long long)cols; ++c)
                    {
                        float v = row[c];
                        facc += v * v;
                    }
                    acc += (double)facc;
                }
                row_sumsq[r] = acc;
            }
            else if (allow_avx2_fallback)
            {
#elif defined(__AVX2__)
            const auto &renv = llaminar::debugEnv().rmsnorm;
            bool allow_avx2 = (renv.vec_impl == 0 || renv.vec_impl == 2);
            if (allow_avx2)
            {
                long long c = 0;
                __m256 acc0 = _mm256_setzero_ps();
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                __m256 acc3 = _mm256_setzero_ps();
                for (; c + 32 <= (long long)cols; c += 32)
                {
                    const float *base = row + c;
                    __m256 v0 = _mm256_loadu_ps(base + 0);
                    __m256 v1 = _mm256_loadu_ps(base + 8);
                    __m256 v2 = _mm256_loadu_ps(base + 16);
                    __m256 v3 = _mm256_loadu_ps(base + 24);
                    acc0 = _mm256_fmadd_ps(v0, v0, acc0);
                    acc1 = _mm256_fmadd_ps(v1, v1, acc1);
                    acc2 = _mm256_fmadd_ps(v2, v2, acc2);
                    acc3 = _mm256_fmadd_ps(v3, v3, acc3);
                }
                __m256 acc01 = _mm256_add_ps(acc0, acc1);
                __m256 acc23 = _mm256_add_ps(acc2, acc3);
                __m256 acc_all = _mm256_add_ps(acc01, acc23);
                // Horizontal reduce 8 floats
                __m128 lo = _mm256_castps256_ps128(acc_all);
                __m128 hi = _mm256_extractf128_ps(acc_all, 1);
                __m128 sum2 = _mm_add_ps(lo, hi);
                __m128 shuf = _mm_movehdup_ps(sum2); // (a1,a1,b1,b1)
                __m128 sums = _mm_add_ps(sum2, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                float partial;
                _mm_store_ss(&partial, sums);
                acc += (double)partial;
                for (; c + 8 <= (long long)cols; c += 8)
                {
                    __m256 v = _mm256_loadu_ps(row + c);
                    __m256 sq = _mm256_mul_ps(v, v);
                    // Reduce 8
                    __m128 lo2 = _mm256_castps256_ps128(sq);
                    __m128 hi2 = _mm256_extractf128_ps(sq, 1);
                    __m128 sumv = _mm_add_ps(lo2, hi2);
                    __m128 sh2 = _mm_movehdup_ps(sumv);
                    __m128 sumv2 = _mm_add_ps(sumv, sh2);
                    sh2 = _mm_movehl_ps(sh2, sumv2);
                    sumv2 = _mm_add_ss(sumv2, sh2);
                    float p;
                    _mm_store_ss(&p, sumv2);
                    acc += (double)p;
                }
                if (!fast_acc)
                {
                    for (; c < (long long)cols; ++c)
                    {
                        float v = row[c];
                        acc += (double)v * (double)v;
                    }
                }
                else
                {
                    float facc_tail = 0.f;
                    for (; c < (long long)cols; ++c)
                    {
                        float v = row[c];
                        facc_tail += v * v;
                    }
                    acc += (double)facc_tail;
                }
                row_sumsq[r] = acc;
            }
            else
            {
#endif // end AVX2 fallback else
                if (!fast_acc)
                {
                    double s_scalar = 0.0;
#pragma omp simd reduction(+ : s_scalar)
                    for (long long c2 = 0; c2 < (long long)cols; ++c2)
                    {
                        double v = (double)row[c2];
                        s_scalar += v * v;
                    }
                    row_sumsq[r] = s_scalar;
                }
                else
                {
                    float facc_scalar = 0.f;
#pragma omp simd reduction(+ : facc_scalar)
                    for (long long c2 = 0; c2 < (long long)cols; ++c2)
                    {
                        float v = row[c2];
                        facc_scalar += v * v;
                    }
                    row_sumsq[r] = (double)facc_scalar;
                }
#if defined(__AVX512F__) || defined(__AVX2__)
            } // end feature gated path selection
#else
                // Pure scalar fallback (no AVX2/AVX512 compiled)
                if (!fast_acc)
                {
                    double s_scalar_fallback = 0.0;
#pragma omp simd reduction(+ : s_scalar_fallback)
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        double v = (double)row[c];
                        s_scalar_fallback += v * v;
                    }
                    row_sumsq[r] = s_scalar_fallback;
                }
                else
                {
                    float facc_scalar_fb = 0.f;
#pragma omp simd reduction(+ : facc_scalar_fb)
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        float v = row[c];
                        facc_scalar_fb += v * v;
                    }
                    row_sumsq[r] = (double)facc_scalar_fb;
                }
#endif // feature blocks
        }
        if (fast_acc && rows > 0)
        {
            const float *row0 = src;
            double ref = 0.0;
            for (size_t c = 0; c < cols; ++c)
            {
                double v = (double)row0[c];
                ref += v * v;
            }
            double diff = std::fabs(ref - row_sumsq[0]);
            double rel = (ref > 0.0) ? diff / ref : 0.0;
            if (rel > 5e-4)
            {
                static bool warned = false;
                if (!warned)
                {
                    warned = true;
                    LOG_WARN("[RMSNorm] fast_accumulate relative deviation=" << rel << " ref=" << ref << " got=" << row_sumsq[0]);
                }
            }
        }
    }

    void rmsnorm_compute_inv(const double *row_sumsq,
                             std::size_t rows,
                             std::size_t cols,
                             float epsilon,
                             float *inv_out)
    {
        if (!row_sumsq || !inv_out || rows == 0)
            return;
        double denom = (double)cols;
        if (denom <= 0.0)
            denom = 1.0;
        for (std::size_t r = 0; r < rows; ++r)
        {
            double val = row_sumsq[r] / denom + (double)epsilon;
            double inv = (val > 0.0) ? 1.0 / std::sqrt(val) : 0.0;
            inv_out[r] = (float)inv;
        }
    }

    void rmsnorm_apply(const float *src,
                       const float *gamma,
                       const float *inv,
                       std::size_t rows,
                       std::size_t cols,
                       float *dst,
                       GammaMode mode,
                       std::size_t gamma_offset,
                       const RMSNormExecOptions &opts)
    {
        if (!src || !inv || !dst || rows == 0 || cols == 0)
            return;
        bool parallel = want_parallel(rows, cols, opts);
        bool has_gamma = (gamma != nullptr);
#pragma omp parallel for if (parallel)
        for (long long r = 0; r < (long long)rows; ++r)
        {
            const float *row_in = src + (std::size_t)r * cols;
            float *row_out = dst + (std::size_t)r * cols;
            float scale = inv[(std::size_t)r];
            if (scale == 0.0f)
            {
                std::fill(row_out, row_out + cols, 0.0f);
                continue;
            }
            if (!has_gamma)
            {
#pragma omp simd
                for (long long c = 0; c < (long long)cols; ++c)
                {
                    row_out[c] = row_in[c] * scale;
                }
            }
            else
            {
                if (mode == GammaMode::REPLICATED)
                {
#pragma omp simd
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        row_out[c] = row_in[c] * scale * gamma[gamma_offset + (std::size_t)c];
                    }
                }
                else
                { // SHARDED
#pragma omp simd
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        row_out[c] = row_in[c] * scale * gamma[c];
                    }
                }
            }
        }
    }

    void rmsnorm_row_major_fused(const float *src,
                                 const float *gamma,
                                 float *dst,
                                 std::size_t rows,
                                 std::size_t cols,
                                 float epsilon,
                                 GammaMode mode,
                                 std::size_t gamma_offset,
                                 const RMSNormExecOptions &opts)
    {
        if (!src || !dst || rows == 0 || cols == 0)
            return;
        auto &env = llaminar::debugEnv().rmsnorm;
        RMSNormExecOptions eff = opts;
        if (env.force_scalar)
        {
            eff.allow_parallel = false;
            eff.force_scalar = true;
        }
        if (env.disable_tls_scratch)
        {
            // Fallback to ephemeral allocations (original behavior)
            std::vector<double> row_sumsq(rows, 0.0);
            std::vector<float> inv(rows, 0.0f);
            rmsnorm_compute_row_sumsq(src, rows, cols, row_sumsq.data(), eff);
            rmsnorm_compute_inv(row_sumsq.data(), rows, cols, epsilon, inv.data());
            rmsnorm_apply(src, gamma, inv.data(), rows, cols, dst, mode, gamma_offset, eff);
            return;
        }
        // Use thread-local scratch
        thread_local RMSNormScratch tls_scratch; // zero-cost after first instantiation
        if (env.scratch_prealloc_rows > 0 && tls_scratch.row_sumsq.capacity() < (size_t)env.scratch_prealloc_rows)
        {
            tls_scratch.row_sumsq.reserve(env.scratch_prealloc_rows);
            tls_scratch.inv.reserve(env.scratch_prealloc_rows);
        }
        tls_scratch.ensure(rows);
        rmsnorm_compute_row_sumsq(src, rows, cols, tls_scratch.row_sumsq.data(), eff);
        rmsnorm_compute_inv(tls_scratch.row_sumsq.data(), rows, cols, epsilon, tls_scratch.inv.data());
        rmsnorm_apply(src, gamma, tls_scratch.inv.data(), rows, cols, dst, mode, gamma_offset, eff);
    }

    void rmsnorm_row_major_fused(const float *src,
                                 const float *gamma,
                                 float *dst,
                                 std::size_t rows,
                                 std::size_t cols,
                                 float epsilon,
                                 RMSNormScratch &scratch,
                                 GammaMode mode,
                                 std::size_t gamma_offset,
                                 const RMSNormExecOptions &opts)
    {
        if (!src || !dst || rows == 0 || cols == 0)
            return;
        scratch.ensure(rows);
        rmsnorm_compute_row_sumsq(src, rows, cols, scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(scratch.row_sumsq.data(), rows, cols, epsilon, scratch.inv.data());
        rmsnorm_apply(src, gamma, scratch.inv.data(), rows, cols, dst, mode, gamma_offset, opts);
    }

} // namespace llaminar::kernels
