/**
 * @file RMSNormPrimitives.cpp
 * @brief Vectorized RMSNorm implementation (ported from V1)
 * @author David Sanftenberg
 */

#include "RMSNormPrimitives.h"
#include "utils/DebugEnv.h"
#include <omp.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../../../utils/OpenMPUtils.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar2::primitives
{

    // --- FP16 Scalar ---
    __attribute__((always_inline)) inline float fp16_to_fp32_scalar(uint16_t h)
    {
        uint32_t sign = (h & 0x8000U) << 16;
        uint32_t exp = (h & 0x7C00U) >> 10;
        uint32_t mantissa = h & 0x03FFU;
        uint32_t fp32_bits;
        if (exp == 0)
        {
            if (mantissa == 0)
                fp32_bits = sign;
            else
            {
                exp = 1;
                while ((mantissa & 0x0400U) == 0)
                {
                    mantissa <<= 1;
                    exp--;
                }
                mantissa &= 0x03FFU;
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
            }
        }
        else if (exp == 0x1F)
        {
            fp32_bits = sign | 0x7F800000U | (mantissa << 13);
        }
        else
        {
            fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &fp32_bits, sizeof(float));
        return result;
    }

    __attribute__((always_inline)) inline uint16_t fp32_to_fp16_scalar(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        uint32_t sign = (bits >> 16) & 0x8000U;
        int32_t exp = ((bits >> 23) & 0xFFU) - 127 + 15;
        uint32_t mantissa = (bits >> 13) & 0x03FFU;
        if (exp <= 0)
        {
            if (exp < -10)
                return sign;
            mantissa |= 0x0400U;
            mantissa >>= (1 - exp);
            return sign | mantissa;
        }
        else if (exp >= 0x1F)
        {
            return sign | 0x7C00U;
        }
        else
        {
            return sign | (exp << 10) | mantissa;
        }
    }

    // --- BF16 Scalar ---
    __attribute__((always_inline)) inline float bf16_to_fp32_scalar(uint16_t h)
    {
        uint32_t bits = static_cast<uint32_t>(h) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(float));
        return f;
    }

    __attribute__((always_inline)) inline uint16_t fp32_to_bf16_scalar(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(float));
        uint32_t lsb = (bits >> 16) & 1;
        uint32_t bias = 0x7FFF + lsb;
        bits += bias;
        return static_cast<uint16_t>(bits >> 16);
    }

#if defined(__AVX512F__)
    // --- AVX512 BF16 ---
    __attribute__((always_inline)) inline __m512 bf16_to_fp32_avx512(const uint16_t *bf16_ptr)
    {
        __m256i bf16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(bf16_ptr));
        __m512i fp32_int = _mm512_cvtepu16_epi32(bf16_vec);
        __m512i shifted = _mm512_slli_epi32(fp32_int, 16);
        return _mm512_castsi512_ps(shifted);
    }

    __attribute__((always_inline)) inline void fp32_to_bf16_avx512(__m512 fp32_vec, uint16_t *bf16_ptr)
    {
        __m512i fp32_int = _mm512_castps_si512(fp32_vec);
        __m512i bits = fp32_int;
        __m512i bit16 = _mm512_srli_epi32(bits, 16);
        __m512i lsb = _mm512_and_si512(bit16, _mm512_set1_epi32(1));
        __m512i bias_base = _mm512_set1_epi32(0x7FFF);
        __m512i bias = _mm512_add_epi32(bias_base, lsb);
        __m512i rounded = _mm512_add_epi32(bits, bias);
        __m512i bf16_32 = _mm512_srli_epi32(rounded, 16);
        __m256i bf16_vec = _mm512_cvtepi32_epi16(bf16_32);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(bf16_ptr), bf16_vec);
    }

    // --- AVX512 FP16 ---
#if defined(__AVX512FP16__)
    __attribute__((always_inline)) inline __m512 fp16_to_fp32_avx512fp16(const uint16_t *fp16_ptr)
    {
        __m256h fp16_vec = _mm256_loadu_ph(reinterpret_cast<const void *>(fp16_ptr));
        return _mm512_cvtph_ps(fp16_vec);
    }

    __attribute__((always_inline)) inline void fp32_to_fp16_avx512fp16(__m512 fp32_vec, uint16_t *fp16_ptr)
    {
        __m256h fp16_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_ph(reinterpret_cast<void *>(fp16_ptr), fp16_vec);
    }
#else
    __attribute__((always_inline)) inline __m512 fp16_to_fp32_avx512(const uint16_t *fp16_ptr)
    {
        __m256i fp16_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(fp16_ptr));
        return _mm512_cvtph_ps(fp16_vec);
    }

    __attribute__((always_inline)) inline void fp32_to_fp16_avx512(__m512 fp32_vec, uint16_t *fp16_ptr)
    {
        __m256i fp16_vec = _mm512_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(fp16_ptr), fp16_vec);
    }
#endif
#endif

#if defined(__AVX2__)
    // --- AVX2 BF16 ---
    __attribute__((always_inline)) inline __m256 bf16_to_fp32_avx2(const uint16_t *bf16_ptr)
    {
        __m128i bf16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(bf16_ptr));
        __m256i fp32_int = _mm256_cvtepu16_epi32(bf16_vec);
        __m256i shifted = _mm256_slli_epi32(fp32_int, 16);
        return _mm256_castsi256_ps(shifted);
    }

    __attribute__((always_inline)) inline void fp32_to_bf16_avx2(__m256 fp32_vec, uint16_t *bf16_ptr)
    {
        __m256i fp32_int = _mm256_castps_si256(fp32_vec);
        __m256i bits = fp32_int;
        __m256i bit16 = _mm256_srli_epi32(bits, 16);
        __m256i lsb = _mm256_and_si256(bit16, _mm256_set1_epi32(1));
        __m256i bias_base = _mm256_set1_epi32(0x7FFF);
        __m256i bias = _mm256_add_epi32(bias_base, lsb);
        __m256i rounded = _mm256_add_epi32(bits, bias);
        __m256i bf16_32 = _mm256_srli_epi32(rounded, 16);
        __m256i packed = _mm256_packus_epi32(bf16_32, _mm256_setzero_si256());
        __m256i permuted = _mm256_permute4x64_epi64(packed, _MM_SHUFFLE(3, 1, 2, 0));
        __m128i result = _mm256_castsi256_si128(permuted);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(bf16_ptr), result);
    }

#if defined(__F16C__)
    // --- AVX2 FP16 ---
    __attribute__((always_inline)) inline __m256 fp16_to_fp32_avx2(const uint16_t *fp16_ptr)
    {
        __m128i fp16_vec = _mm_loadu_si128(reinterpret_cast<const __m128i *>(fp16_ptr));
        return _mm256_cvtph_ps(fp16_vec);
    }

    __attribute__((always_inline)) inline void fp32_to_fp16_avx2(__m256 fp32_vec, uint16_t *fp16_ptr)
    {
        __m128i fp16_vec = _mm256_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(fp16_ptr), fp16_vec);
    }
#endif
#endif

    namespace
    {
        inline bool want_parallel(std::size_t rows, std::size_t cols, const RMSNormExecOptions &opts)
        {
            if (!opts.allow_parallel)
                return false;

            std::size_t elems = rows * cols;

            // Single-row decode for large models (d_model >= 2048)
            // If rows=1, we want parallel if cols is large enough.
            if (rows == 1)
            {
                // Require significant work per thread to justify parallelization overhead.
                // For 3584 elements, sequential is likely faster than spawning threads.
                return cols >= 8192;
            }

            // NOTE: With OMP_WORKSHARE_REGION, we don't need to check omp_in_parallel()
            // here because the macro handles nested parallelism correctly - it will use
            // worksharing constructs when inside an existing parallel region.

            return elems >= opts.parallel_threshold_elems;
        }

#if defined(__AVX512F__)
        __attribute__((always_inline)) inline double compute_sumsq_avx512(const float *row, std::size_t cols)
        {
            double acc = 0.0;
            long long c = 0;
            __m512d dacc0 = _mm512_setzero_pd();
            __m512d dacc1 = _mm512_setzero_pd();
            __m512d dacc2 = _mm512_setzero_pd();
            __m512d dacc3 = _mm512_setzero_pd();

            for (; c + 64 <= (long long)cols; c += 64)
            {
                const float *base = row + c;

                // Load 4x 16-float blocks
                __m512 v0 = _mm512_loadu_ps(base + 0);
                __m512 v1 = _mm512_loadu_ps(base + 16);
                __m512 v2 = _mm512_loadu_ps(base + 32);
                __m512 v3 = _mm512_loadu_ps(base + 48);

                // Convert to double (split each 16-float into 2x 8-double)
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

                // FMA: acc += d * d
                // Distribute across 4 accumulators to break dependency chains
                dacc0 = _mm512_fmadd_pd(d0a, d0a, dacc0);
                dacc1 = _mm512_fmadd_pd(d0b, d0b, dacc1);
                dacc2 = _mm512_fmadd_pd(d1a, d1a, dacc2);
                dacc3 = _mm512_fmadd_pd(d1b, d1b, dacc3);

                dacc0 = _mm512_fmadd_pd(d2a, d2a, dacc0);
                dacc1 = _mm512_fmadd_pd(d2b, d2b, dacc1);
                dacc2 = _mm512_fmadd_pd(d3a, d3a, dacc2);
                dacc3 = _mm512_fmadd_pd(d3b, d3b, dacc3);
            }

            // Horizontal reduction
            acc += _mm512_reduce_add_pd(dacc0);
            acc += _mm512_reduce_add_pd(dacc1);
            acc += _mm512_reduce_add_pd(dacc2);
            acc += _mm512_reduce_add_pd(dacc3);

            // Tail: 16-float blocks
            for (; c + 16 <= (long long)cols; c += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c);
                __m256 vlo = _mm512_castps512_ps256(v);
                __m256 vhi = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(v), 1));

                __m512d d0 = _mm512_cvtps_pd(vlo);
                __m512d d1 = _mm512_cvtps_pd(vhi);

                __m512d sq0 = _mm512_mul_pd(d0, d0);
                __m512d sq1 = _mm512_mul_pd(d1, d1);

                acc += _mm512_reduce_add_pd(sq0);
                acc += _mm512_reduce_add_pd(sq1);
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }
            return acc;
        }
#elif defined(__AVX2__)
        __attribute__((always_inline)) inline double compute_sumsq_avx2(const float *row, std::size_t cols)
        {
            double acc = 0.0;
            long long c = 0;
            __m256d dacc0 = _mm256_setzero_pd();
            __m256d dacc1 = _mm256_setzero_pd();
            __m256d dacc2 = _mm256_setzero_pd();
            __m256d dacc3 = _mm256_setzero_pd();

            for (; c + 32 <= (long long)cols; c += 32)
            {
                const float *base = row + c;
                __m256 v0 = _mm256_loadu_ps(base + 0);
                __m256 v1 = _mm256_loadu_ps(base + 8);
                __m256 v2 = _mm256_loadu_ps(base + 16);
                __m256 v3 = _mm256_loadu_ps(base + 24);

                // Convert v0
                __m128 v0_lo = _mm256_castps256_ps128(v0);
                __m128 v0_hi = _mm256_extractf128_ps(v0, 1);
                __m256d d0_lo = _mm256_cvtps_pd(v0_lo);
                __m256d d0_hi = _mm256_cvtps_pd(v0_hi);

                // Convert v1
                __m128 v1_lo = _mm256_castps256_ps128(v1);
                __m128 v1_hi = _mm256_extractf128_ps(v1, 1);
                __m256d d1_lo = _mm256_cvtps_pd(v1_lo);
                __m256d d1_hi = _mm256_cvtps_pd(v1_hi);

                // Convert v2
                __m128 v2_lo = _mm256_castps256_ps128(v2);
                __m128 v2_hi = _mm256_extractf128_ps(v2, 1);
                __m256d d2_lo = _mm256_cvtps_pd(v2_lo);
                __m256d d2_hi = _mm256_cvtps_pd(v2_hi);

                // Convert v3
                __m128 v3_lo = _mm256_castps256_ps128(v3);
                __m128 v3_hi = _mm256_extractf128_ps(v3, 1);
                __m256d d3_lo = _mm256_cvtps_pd(v3_lo);
                __m256d d3_hi = _mm256_cvtps_pd(v3_hi);

                dacc0 = _mm256_fmadd_pd(d0_lo, d0_lo, dacc0);
                dacc1 = _mm256_fmadd_pd(d0_hi, d0_hi, dacc1);
                dacc2 = _mm256_fmadd_pd(d1_lo, d1_lo, dacc2);
                dacc3 = _mm256_fmadd_pd(d1_hi, d1_hi, dacc3);

                dacc0 = _mm256_fmadd_pd(d2_lo, d2_lo, dacc0);
                dacc1 = _mm256_fmadd_pd(d2_hi, d2_hi, dacc1);
                dacc2 = _mm256_fmadd_pd(d3_lo, d3_lo, dacc2);
                dacc3 = _mm256_fmadd_pd(d3_hi, d3_hi, dacc3);
            }

            // Combine accumulators
            __m256d dacc01 = _mm256_add_pd(dacc0, dacc1);
            __m256d dacc23 = _mm256_add_pd(dacc2, dacc3);
            __m256d dacc_all = _mm256_add_pd(dacc01, dacc23);

            // Horizontal reduction
            __m128d lo = _mm256_castpd256_pd128(dacc_all);
            __m128d hi = _mm256_extractf128_pd(dacc_all, 1);
            __m128d sum2 = _mm_add_pd(lo, hi);
            __m128d swap = _mm_permute_pd(sum2, 1);
            __m128d sum1 = _mm_add_pd(sum2, swap);

            double partial;
            _mm_store_sd(&partial, sum1);
            acc += partial;

            // 8-element blocks
            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c);
                __m128 v_lo = _mm256_castps256_ps128(v);
                __m128 v_hi = _mm256_extractf128_ps(v, 1);

                __m256d d_lo = _mm256_cvtps_pd(v_lo);
                __m256d d_hi = _mm256_cvtps_pd(v_hi);

                __m256d sq_lo = _mm256_mul_pd(d_lo, d_lo);
                __m256d sq_hi = _mm256_mul_pd(d_hi, d_hi);

                __m256d sum_blk = _mm256_add_pd(sq_lo, sq_hi);

                __m128d l = _mm256_castpd256_pd128(sum_blk);
                __m128d h = _mm256_extractf128_pd(sum_blk, 1);
                __m128d s = _mm_add_pd(l, h);
                __m128d sw = _mm_permute_pd(s, 1);
                __m128d s1 = _mm_add_pd(s, sw);

                double p;
                _mm_store_sd(&p, s1);
                acc += p;
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }
            return acc;
        }
#endif

        __attribute__((always_inline)) inline double compute_sumsq_scalar(const float *row, std::size_t cols)
        {
            double s_scalar = 0.0;
#pragma omp simd reduction(+ : s_scalar)
            for (long long c = 0; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                s_scalar += v * v;
            }
            return s_scalar;
        }

        __attribute__((always_inline)) inline double compute_sumsq_dispatch(const float *row, std::size_t cols)
        {
#if defined(__AVX512F__)
            return compute_sumsq_avx512(row, cols);
#elif defined(__AVX2__)
            return compute_sumsq_avx2(row, cols);
#else
            return compute_sumsq_scalar(row, cols);
#endif
        }

    } // anonymous namespace

    void rmsnorm_compute_row_sumsq_vectorized(
        const float *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts)
    {
        if (!src || !row_sumsq || rows == 0 || cols == 0)
            return;

        // T5 compatibility: float32 accumulation
        if (opts.t5_compat_mode)
        {
            bool parallel = want_parallel(rows, cols, opts);
            if (parallel)
            {
                auto do_work = [&]()
                {
#pragma omp for schedule(static)
                    for (long long r = 0; r < (long long)rows; ++r)
                    {
                        const float *row = src + (std::size_t)r * cols;
                        float sum_sq = 0.0f;
                        for (std::size_t c = 0; c < cols; ++c)
                        {
                            float val = row[c];
                            sum_sq += val * val;
                        }
                        row_sumsq[r] = (double)sum_sq;
                    }
                };
                OMP_WORKSHARE_REGION(do_work);
            }
            else
            {
                // Serial path
                for (long long r = 0; r < (long long)rows; ++r)
                {
                    const float *row = src + (std::size_t)r * cols;
                    float sum_sq = 0.0f;
                    for (std::size_t c = 0; c < cols; ++c)
                    {
                        float val = row[c];
                        sum_sq += val * val;
                    }
                    row_sumsq[r] = (double)sum_sq;
                }
            }
            return;
        }

        // Double precision vectorized path
        bool parallel = want_parallel(rows, cols, opts);

        // Special case: Single row with large columns -> Parallelize reduction
        if (rows == 1 && parallel)
        {
            double total_sum_sq = 0.0;

            // Note: For reduction in nested context, we use OMP_WORKSHARE_REGION
            // but need manual chunking since omp for reduction can be tricky
            auto do_reduction = [&]()
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();

                long long chunk_size = (cols + nthreads - 1) / nthreads;
                long long start = tid * chunk_size;
                long long end = std::min((long long)cols, start + chunk_size);

                double local_sum = 0.0;
                if (start < end)
                {
                    local_sum = compute_sumsq_dispatch(src + start, end - start);
                }

// Thread-safe accumulation
#pragma omp atomic
                total_sum_sq += local_sum;
            };
            OMP_WORKSHARE_REGION(do_reduction);
            row_sumsq[0] = total_sum_sq;
            return;
        }

        if (parallel)
        {
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (long long r = 0; r < (long long)rows; ++r)
                {
                    const float *row = src + (std::size_t)r * cols;
                    row_sumsq[r] = compute_sumsq_dispatch(row, cols);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path
            for (long long r = 0; r < (long long)rows; ++r)
            {
                const float *row = src + (std::size_t)r * cols;
                row_sumsq[r] = compute_sumsq_dispatch(row, cols);
            }
        }
    }

    void rmsnorm_compute_inv(
        const double *row_sumsq,
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

    void rmsnorm_apply_vectorized(
        const float *src,
        const float *gamma,
        const float *inv,
        std::size_t rows,
        std::size_t cols,
        float *dst,
        const RMSNormExecOptions &opts)
    {
        if (!src || !inv || !dst || rows == 0 || cols == 0)
            return;

        bool parallel = want_parallel(rows, cols, opts);
        bool has_gamma = (gamma != nullptr);

        // Special case: Single row with large columns -> Parallelize application
        if (rows == 1 && parallel)
        {
            float scale = inv[0];
            if (scale == 0.0f)
            {
                std::fill(dst, dst + cols, 0.0f);
                return;
            }

            if (has_gamma)
            {
                auto do_work = [&]()
                {
#pragma omp for schedule(static)
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        dst[c] = src[c] * scale * gamma[c];
                    }
                };
                OMP_WORKSHARE_REGION(do_work);
            }
            else
            {
                auto do_work = [&]()
                {
#pragma omp for schedule(static)
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        dst[c] = src[c] * scale;
                    }
                };
                OMP_WORKSHARE_REGION(do_work);
            }
            return;
        }

        if (parallel)
        {
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
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
#pragma omp simd
                        for (long long c = 0; c < (long long)cols; ++c)
                        {
                            row_out[c] = row_in[c] * scale * gamma[c];
                        }
                    }
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path
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
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        row_out[c] = row_in[c] * scale;
                    }
                }
                else
                {
                    for (long long c = 0; c < (long long)cols; ++c)
                    {
                        row_out[c] = row_in[c] * scale * gamma[c];
                    }
                }
            }
        }
    }

    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        thread_local RMSNormScratch tls_scratch;
        tls_scratch.ensure(rows);

        rmsnorm_compute_row_sumsq_vectorized(src, rows, cols, tls_scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(tls_scratch.row_sumsq.data(), rows, cols, epsilon, tls_scratch.inv.data());
        rmsnorm_apply_vectorized(src, gamma, tls_scratch.inv.data(), rows, cols, dst, opts);
    }

    void rmsnorm_fused_vectorized(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        RMSNormScratch &scratch,
        const RMSNormExecOptions &opts)
    {
        scratch.ensure(rows);
        rmsnorm_compute_row_sumsq_vectorized(src, rows, cols, scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(scratch.row_sumsq.data(), rows, cols, epsilon, scratch.inv.data());
        rmsnorm_apply_vectorized(src, gamma, scratch.inv.data(), rows, cols, dst, opts);
    }

    // ========================================================================
    // INT32 RMSNorm Implementation (for full INT8 pipelines)
    // ========================================================================

    void rmsnorm_compute_row_sumsq_int32_vectorized(
        const int32_t *src,
        std::size_t rows,
        std::size_t cols,
        double *row_sumsq,
        const RMSNormExecOptions &opts)
    {
        if (!src || !row_sumsq || rows == 0 || cols == 0)
            return;

        bool parallel = want_parallel(rows, cols, opts);

        auto do_row_work = [&](long long r)
        {
            const int32_t *row = src + (std::size_t)r * cols;
            double acc = 0.0;

#if defined(__AVX512F__)
            // AVX512: Process 16 INT32 values per iteration (double accumulation)
            long long c = 0;
            __m512d dacc0 = _mm512_setzero_pd();
            __m512d dacc1 = _mm512_setzero_pd();

            for (; c + 16 <= (long long)cols; c += 16)
            {
                // Load 16 INT32 values
                __m512i i32_lo = _mm512_loadu_si512((__m512i *)(row + c)); // 0-15

                // Convert INT32 to double (8 at a time)
                __m256i i32_lo_256 = _mm512_castsi512_si256(i32_lo);
                __m256i i32_hi_256 = _mm512_extracti64x4_epi64(i32_lo, 1);

                __m512d d0 = _mm512_cvtepi32_pd(i32_lo_256); // 0-7
                __m512d d1 = _mm512_cvtepi32_pd(i32_hi_256); // 8-15

                // FMA: acc += d * d
                dacc0 = _mm512_fmadd_pd(d0, d0, dacc0);
                dacc1 = _mm512_fmadd_pd(d1, d1, dacc1);
            }

            // Horizontal reduction
            acc += _mm512_reduce_add_pd(dacc0);
            acc += _mm512_reduce_add_pd(dacc1);

            // Tail: 8-int32 blocks
            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256i i32 = _mm256_loadu_si256((__m256i *)(row + c));
                __m512d d = _mm512_cvtepi32_pd(i32);
                __m512d sq = _mm512_mul_pd(d, d);
                acc += _mm512_reduce_add_pd(sq);
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }

#elif defined(__AVX2__)
            // AVX2: Process 16 INT32 values per iteration (FP32 accumulation, then convert to double)
            long long c = 0;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (; c + 16 <= (long long)cols; c += 16)
            {
                // Load 16 INT32 values
                __m256i i32_0 = _mm256_loadu_si256((__m256i *)(row + c));
                __m256i i32_1 = _mm256_loadu_si256((__m256i *)(row + c + 8));

                // Convert INT32 to FP32
                __m256 f0 = _mm256_cvtepi32_ps(i32_0);
                __m256 f1 = _mm256_cvtepi32_ps(i32_1);

                // FMA: acc += f * f
                acc0 = _mm256_fmadd_ps(f0, f0, acc0);
                acc1 = _mm256_fmadd_ps(f1, f1, acc1);
            }

            // Combine accumulators and reduce to scalar
            __m256 acc_all = _mm256_add_ps(acc0, acc1);

            // Horizontal reduction
            __m128 lo = _mm256_castps256_ps128(acc_all);
            __m128 hi = _mm256_extractf128_ps(acc_all, 1);
            __m128 sum2 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum2);
            __m128 sums = _mm_add_ps(sum2, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);

            float partial;
            _mm_store_ss(&partial, sums);
            acc += partial;

            // Tail: 8-element blocks
            for (; c + 8 <= (long long)cols; c += 8)
            {
                __m256i i32 = _mm256_loadu_si256((__m256i *)(row + c));
                __m256 f = _mm256_cvtepi32_ps(i32);
                __m256 sq = _mm256_mul_ps(f, f);

                __m128 lo2 = _mm256_castps256_ps128(sq);
                __m128 hi2 = _mm256_extractf128_ps(sq, 1);
                __m128 sumv = _mm_add_ps(lo2, hi2);
                __m128 sh2 = _mm_movehdup_ps(sumv);
                __m128 sumv2 = _mm_add_ps(sumv, sh2);
                sh2 = _mm_movehl_ps(sh2, sumv2);
                sumv2 = _mm_add_ss(sumv2, sh2);

                float p;
                _mm_store_ss(&p, sumv2);
                acc += p;
            }

            // Scalar tail
            for (; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                acc += v * v;
            }

#else
            // Scalar fallback
            double s_scalar = 0.0;
#pragma omp simd reduction(+ : s_scalar)
            for (long long c = 0; c < (long long)cols; ++c)
            {
                double v = (double)row[c];
                s_scalar += v * v;
            }
            acc = s_scalar;
#endif

            row_sumsq[r] = acc;
        };

        if (parallel)
        {
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (long long r = 0; r < (long long)rows; ++r)
                {
                    do_row_work(r);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path
            for (long long r = 0; r < (long long)rows; ++r)
            {
                do_row_work(r);
            }
        }
    }

    namespace
    {
        // ========== INT32→INT8 Helper Kernels ==========

        /// INT32→FP32 normalization - Scalar path
        __attribute__((always_inline)) inline void int32_to_fp32_normalize_scalar(
            const int32_t *src,
            float *dst,
            float rms_inv,
            const float *gamma,
            std::size_t cols)
        {
            if (gamma)
            {
                for (std::size_t c = 0; c < cols; ++c)
                {
                    dst[c] = static_cast<float>(src[c]) * rms_inv * gamma[c];
                }
            }
            else
            {
                for (std::size_t c = 0; c < cols; ++c)
                {
                    dst[c] = static_cast<float>(src[c]) * rms_inv;
                }
            }
        }

#if defined(__AVX512F__)
        /// INT32→FP32 normalization - AVX512 path
        __attribute__((always_inline)) inline void int32_to_fp32_normalize_avx512(
            const int32_t *src,
            float *dst,
            float rms_inv,
            const float *gamma,
            std::size_t cols)
        {
            __m512 rms_inv_vec = _mm512_set1_ps(rms_inv);
            std::size_t c = 0;

            if (gamma)
            {
                for (; c + 16 <= cols; c += 16)
                {
                    __m512i i32_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + c));
                    __m512 fp32_vec = _mm512_cvtepi32_ps(i32_vec);
                    __m512 gamma_vec = _mm512_loadu_ps(gamma + c);
                    __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(fp32_vec, rms_inv_vec), gamma_vec);
                    _mm512_storeu_ps(dst + c, normalized);
                }
            }
            else
            {
                for (; c + 16 <= cols; c += 16)
                {
                    __m512i i32_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(src + c));
                    __m512 fp32_vec = _mm512_cvtepi32_ps(i32_vec);
                    __m512 normalized = _mm512_mul_ps(fp32_vec, rms_inv_vec);
                    _mm512_storeu_ps(dst + c, normalized);
                }
            }

            // Scalar tail
            int32_to_fp32_normalize_scalar(src + c, dst + c, rms_inv, gamma ? gamma + c : nullptr, cols - c);
        }

        /// Find max absolute value - AVX512 path
        __attribute__((always_inline)) inline float find_max_abs_avx512(const float *data, std::size_t size)
        {
            __m512 max_vec = _mm512_setzero_ps();
            std::size_t i = 0;

            for (; i + 16 <= size; i += 16)
            {
                __m512 vec = _mm512_loadu_ps(data + i);
                __m512 abs_vec = _mm512_abs_ps(vec); // AVX512F has abs
                max_vec = _mm512_max_ps(max_vec, abs_vec);
            }

            // Horizontal max reduction
            float max_val = _mm512_reduce_max_ps(max_vec);

            // Scalar tail
            for (; i < size; ++i)
            {
                max_val = std::max(max_val, std::fabs(data[i]));
            }

            return max_val;
        }

        /// FP32→INT8 quantization - AVX512 path
        __attribute__((always_inline)) inline void fp32_to_int8_quantize_avx512(
            const float *src,
            int8_t *dst,
            float inv_scale,
            std::size_t cols)
        {
            __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);
            __m512 min_val = _mm512_set1_ps(-127.0f);
            __m512 max_val = _mm512_set1_ps(127.0f);
            std::size_t c = 0;

            // Process 16 FP32 → 16 INT8 per iteration
            for (; c + 16 <= cols; c += 16)
            {
                __m512 fp32_vec = _mm512_loadu_ps(src + c);
                __m512 scaled = _mm512_mul_ps(fp32_vec, inv_scale_vec);

                // Clamp to [-127, 127]
                scaled = _mm512_max_ps(scaled, min_val);
                scaled = _mm512_min_ps(scaled, max_val);

                // Round to nearest and convert to INT32
                __m512i i32_vec = _mm512_cvtps_epi32(scaled);

                // Saturate to INT8 range (AVX512BW has efficient pack)
                __m128i i8_vec = _mm512_cvtsepi32_epi8(i32_vec);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst + c), i8_vec);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float scaled = src[c] * inv_scale;
                scaled = std::max(-127.0f, std::min(127.0f, scaled));
                dst[c] = static_cast<int8_t>(std::round(scaled));
            }
        }

#elif defined(__AVX2__)
        /// INT32→FP32 normalization - AVX2 path
        __attribute__((always_inline)) inline void int32_to_fp32_normalize_avx2(
            const int32_t *src,
            float *dst,
            float rms_inv,
            const float *gamma,
            std::size_t cols)
        {
            __m256 rms_inv_vec = _mm256_set1_ps(rms_inv);
            std::size_t c = 0;

            if (gamma)
            {
                for (; c + 8 <= cols; c += 8)
                {
                    __m256i i32_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + c));
                    __m256 fp32_vec = _mm256_cvtepi32_ps(i32_vec);
                    __m256 gamma_vec = _mm256_loadu_ps(gamma + c);
                    __m256 normalized = _mm256_mul_ps(_mm256_mul_ps(fp32_vec, rms_inv_vec), gamma_vec);
                    _mm256_storeu_ps(dst + c, normalized);
                }
            }
            else
            {
                for (; c + 8 <= cols; c += 8)
                {
                    __m256i i32_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(src + c));
                    __m256 fp32_vec = _mm256_cvtepi32_ps(i32_vec);
                    __m256 normalized = _mm256_mul_ps(fp32_vec, rms_inv_vec);
                    _mm256_storeu_ps(dst + c, normalized);
                }
            }

            // Scalar tail
            int32_to_fp32_normalize_scalar(src + c, dst + c, rms_inv, gamma ? gamma + c : nullptr, cols - c);
        }

        /// Find max absolute value - AVX2 path
        __attribute__((always_inline)) inline float find_max_abs_avx2(const float *data, std::size_t size)
        {
            __m256 max_vec = _mm256_setzero_ps();
            __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            std::size_t i = 0;

            for (; i + 8 <= size; i += 8)
            {
                __m256 vec = _mm256_loadu_ps(data + i);
                __m256 abs_vec = _mm256_and_ps(vec, sign_mask);
                max_vec = _mm256_max_ps(max_vec, abs_vec);
            }

            // Horizontal max reduction
            __m128 lo = _mm256_castps256_ps128(max_vec);
            __m128 hi = _mm256_extractf128_ps(max_vec, 1);
            __m128 max4 = _mm_max_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(max4);
            __m128 maxs = _mm_max_ps(max4, shuf);
            shuf = _mm_movehl_ps(shuf, maxs);
            maxs = _mm_max_ss(maxs, shuf);

            float max_val;
            _mm_store_ss(&max_val, maxs);

            // Scalar tail
            for (; i < size; ++i)
            {
                max_val = std::max(max_val, std::fabs(data[i]));
            }

            return max_val;
        }

        /// FP32→INT8 quantization - AVX2 path
        __attribute__((always_inline)) inline void fp32_to_int8_quantize_avx2(
            const float *src,
            int8_t *dst,
            float inv_scale,
            std::size_t cols)
        {
            __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);
            __m256 min_val = _mm256_set1_ps(-127.0f);
            __m256 max_val = _mm256_set1_ps(127.0f);
            std::size_t c = 0;

            // Process 8 FP32 → 8 INT8 per iteration
            for (; c + 8 <= cols; c += 8)
            {
                __m256 fp32_vec = _mm256_loadu_ps(src + c);
                __m256 scaled = _mm256_mul_ps(fp32_vec, inv_scale_vec);

                // Clamp to [-127, 127]
                scaled = _mm256_max_ps(scaled, min_val);
                scaled = _mm256_min_ps(scaled, max_val);

                // Round to nearest and convert to INT32
                __m256i i32_vec = _mm256_cvtps_epi32(scaled);

                // Pack INT32 to INT8 (with saturation)
                // AVX2 doesn't have direct INT32→INT8, need multi-step
                __m128i i32_lo = _mm256_castsi256_si128(i32_vec);
                __m128i i32_hi = _mm256_extracti128_si256(i32_vec, 1);
                __m128i i16_packed = _mm_packs_epi32(i32_lo, i32_hi);
                __m128i i8_packed = _mm_packs_epi16(i16_packed, i16_packed);

                _mm_storel_epi64(reinterpret_cast<__m128i *>(dst + c), i8_packed);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float scaled = src[c] * inv_scale;
                scaled = std::max(-127.0f, std::min(127.0f, scaled));
                dst[c] = static_cast<int8_t>(std::round(scaled));
            }
        }
#endif

    } // anonymous namespace

    void rmsnorm_apply_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        const float *inv,
        std::size_t rows,
        std::size_t cols,
        int8_t *dst_int8,
        float *dst_row_scales,
        const RMSNormExecOptions &opts)
    {
        if (!src || !inv || !dst_int8 || !dst_row_scales || rows == 0 || cols == 0)
            return;

        bool parallel = want_parallel(rows, cols, opts);

        auto do_row_work = [&](long long r)
        {
            const int32_t *row_in = src + (std::size_t)r * cols;
            int8_t *row_out = dst_int8 + (std::size_t)r * cols;
            float rms_inv = inv[(std::size_t)r];

            if (rms_inv == 0.0f)
            {
                std::fill(row_out, row_out + cols, (int8_t)0);
                dst_row_scales[r] = 1.0f;
                return;
            }

            // Step 1: Normalize INT32 to FP32 and apply gamma (vectorized)
            thread_local std::vector<float> fp32_buffer;
            if (fp32_buffer.size() < cols)
                fp32_buffer.resize(cols);

#if defined(__AVX512F__)
            int32_to_fp32_normalize_avx512(row_in, fp32_buffer.data(), rms_inv, gamma, cols);
#elif defined(__AVX2__)
            int32_to_fp32_normalize_avx2(row_in, fp32_buffer.data(), rms_inv, gamma, cols);
#else
            int32_to_fp32_normalize_scalar(row_in, fp32_buffer.data(), rms_inv, gamma, cols);
#endif

            // Step 2: Find max absolute value (vectorized)
            float max_abs;
#if defined(__AVX512F__)
            max_abs = find_max_abs_avx512(fp32_buffer.data(), cols);
#elif defined(__AVX2__)
            max_abs = find_max_abs_avx2(fp32_buffer.data(), cols);
#else
            max_abs = 0.0f;
            for (std::size_t c = 0; c < cols; ++c)
            {
                max_abs = std::max(max_abs, std::fabs(fp32_buffer[c]));
            }
#endif

            // Compute scale to fit into INT8 range [-127, 127]
            float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
            dst_row_scales[r] = scale;

            // Step 3: Quantize FP32 to INT8 (vectorized)
            float inv_scale = 1.0f / scale;
#if defined(__AVX512F__)
            fp32_to_int8_quantize_avx512(fp32_buffer.data(), row_out, inv_scale, cols);
#elif defined(__AVX2__)
            fp32_to_int8_quantize_avx2(fp32_buffer.data(), row_out, inv_scale, cols);
#else
            for (std::size_t c = 0; c < cols; ++c)
            {
                float scaled = fp32_buffer[c] * inv_scale;
                scaled = std::max(-127.0f, std::min(127.0f, scaled));
                row_out[c] = static_cast<int8_t>(std::round(scaled));
            }
#endif
        };

        if (parallel)
        {
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (long long r = 0; r < (long long)rows; ++r)
                {
                    do_row_work(r);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            // Serial path
            for (long long r = 0; r < (long long)rows; ++r)
            {
                do_row_work(r);
            }
        }
    }

    void rmsnorm_fused_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        int8_t *dst_int8,
        float *dst_row_scales,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        thread_local RMSNormScratch tls_scratch;
        tls_scratch.ensure(rows);

        rmsnorm_compute_row_sumsq_int32_vectorized(src, rows, cols, tls_scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(tls_scratch.row_sumsq.data(), rows, cols, epsilon, tls_scratch.inv.data());
        rmsnorm_apply_int32_to_int8_vectorized(src, gamma, tls_scratch.inv.data(), rows, cols, dst_int8, dst_row_scales, opts);
    }

    void rmsnorm_fused_int32_to_int8_vectorized(
        const int32_t *src,
        const float *gamma,
        int8_t *dst_int8,
        float *dst_row_scales,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        RMSNormScratch &scratch,
        const RMSNormExecOptions &opts)
    {
        scratch.ensure(rows);
        rmsnorm_compute_row_sumsq_int32_vectorized(src, rows, cols, scratch.row_sumsq.data(), opts);
        rmsnorm_compute_inv(scratch.row_sumsq.data(), rows, cols, epsilon, scratch.inv.data());
        rmsnorm_apply_int32_to_int8_vectorized(src, gamma, scratch.inv.data(), rows, cols, dst_int8, dst_row_scales, opts);
    }

    // ========================================================================
    // FP32 Per-Row Primitives (Testable SIMD variants)
    // ========================================================================

    void rmsnorm_fused_row_scalar(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon)
    {
        // Phase 1: Compute sum of squares (double precision)
        double sum_sq = 0.0;
        for (std::size_t c = 0; c < cols; ++c)
        {
            double val = static_cast<double>(src[c]);
            sum_sq += val * val;
        }

        // Phase 2: Compute inverse RMS
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;

        // Phase 3: Apply normalization
        for (std::size_t c = 0; c < cols; ++c)
        {
            dst[c] = src[c] * inv_rms * gamma[c];
        }
    }

#if defined(__AVX2__)
    void rmsnorm_fused_row_avx2(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon)
    {
        // Phase 1: Compute sum of squares with AVX2
        double sum_sq = 0.0;
        std::size_t c = 0;

        // Multi-accumulator AVX2: process 64 elements per iteration
        __m256d dacc0 = _mm256_setzero_pd();
        __m256d dacc1 = _mm256_setzero_pd();
        __m256d dacc2 = _mm256_setzero_pd();
        __m256d dacc3 = _mm256_setzero_pd();

        for (; c + 64 <= cols; c += 64)
        {
            const float *base = src + c;

            // Load 4x 16-float blocks
            __m256 v0 = _mm256_loadu_ps(base + 0);
            __m256 v1 = _mm256_loadu_ps(base + 8);
            __m256 v2 = _mm256_loadu_ps(base + 16);
            __m256 v3 = _mm256_loadu_ps(base + 24);
            __m256 v4 = _mm256_loadu_ps(base + 32);
            __m256 v5 = _mm256_loadu_ps(base + 40);
            __m256 v6 = _mm256_loadu_ps(base + 48);
            __m256 v7 = _mm256_loadu_ps(base + 56);

            // Convert to double (split each 8-float into 2x 4-double)
            __m128 lo0 = _mm256_castps256_ps128(v0);
            __m128 hi0 = _mm256_extractf128_ps(v0, 1);
            __m128 lo1 = _mm256_castps256_ps128(v1);
            __m128 hi1 = _mm256_extractf128_ps(v1, 1);
            __m128 lo2 = _mm256_castps256_ps128(v2);
            __m128 hi2 = _mm256_extractf128_ps(v2, 1);
            __m128 lo3 = _mm256_castps256_ps128(v3);
            __m128 hi3 = _mm256_extractf128_ps(v3, 1);
            __m128 lo4 = _mm256_castps256_ps128(v4);
            __m128 hi4 = _mm256_extractf128_ps(v4, 1);
            __m128 lo5 = _mm256_castps256_ps128(v5);
            __m128 hi5 = _mm256_extractf128_ps(v5, 1);
            __m128 lo6 = _mm256_castps256_ps128(v6);
            __m128 hi6 = _mm256_extractf128_ps(v6, 1);
            __m128 lo7 = _mm256_castps256_ps128(v7);
            __m128 hi7 = _mm256_extractf128_ps(v7, 1);

            __m256d d0 = _mm256_cvtps_pd(lo0);
            __m256d d1 = _mm256_cvtps_pd(hi0);
            __m256d d2 = _mm256_cvtps_pd(lo1);
            __m256d d3 = _mm256_cvtps_pd(hi1);
            __m256d d4 = _mm256_cvtps_pd(lo2);
            __m256d d5 = _mm256_cvtps_pd(hi2);
            __m256d d6 = _mm256_cvtps_pd(lo3);
            __m256d d7 = _mm256_cvtps_pd(hi3);
            __m256d d8 = _mm256_cvtps_pd(lo4);
            __m256d d9 = _mm256_cvtps_pd(hi4);
            __m256d d10 = _mm256_cvtps_pd(lo5);
            __m256d d11 = _mm256_cvtps_pd(hi5);
            __m256d d12 = _mm256_cvtps_pd(lo6);
            __m256d d13 = _mm256_cvtps_pd(hi6);
            __m256d d14 = _mm256_cvtps_pd(lo7);
            __m256d d15 = _mm256_cvtps_pd(hi7);

            // FMA: acc += d * d
            dacc0 = _mm256_fmadd_pd(d0, d0, dacc0);
            dacc0 = _mm256_fmadd_pd(d1, d1, dacc0);
            dacc0 = _mm256_fmadd_pd(d2, d2, dacc0);
            dacc0 = _mm256_fmadd_pd(d3, d3, dacc0);
            dacc1 = _mm256_fmadd_pd(d4, d4, dacc1);
            dacc1 = _mm256_fmadd_pd(d5, d5, dacc1);
            dacc1 = _mm256_fmadd_pd(d6, d6, dacc1);
            dacc1 = _mm256_fmadd_pd(d7, d7, dacc1);
            dacc2 = _mm256_fmadd_pd(d8, d8, dacc2);
            dacc2 = _mm256_fmadd_pd(d9, d9, dacc2);
            dacc2 = _mm256_fmadd_pd(d10, d10, dacc2);
            dacc2 = _mm256_fmadd_pd(d11, d11, dacc2);
            dacc3 = _mm256_fmadd_pd(d12, d12, dacc3);
            dacc3 = _mm256_fmadd_pd(d13, d13, dacc3);
            dacc3 = _mm256_fmadd_pd(d14, d14, dacc3);
            dacc3 = _mm256_fmadd_pd(d15, d15, dacc3);
        }

        // Horizontal reduction
        __m128d h0 = _mm256_castpd256_pd128(dacc0);
        __m128d h1 = _mm256_extractf128_pd(dacc0, 1);
        __m128d h2 = _mm256_castpd256_pd128(dacc1);
        __m128d h3 = _mm256_extractf128_pd(dacc1, 1);
        __m128d h4 = _mm256_castpd256_pd128(dacc2);
        __m128d h5 = _mm256_extractf128_pd(dacc2, 1);
        __m128d h6 = _mm256_castpd256_pd128(dacc3);
        __m128d h7 = _mm256_extractf128_pd(dacc3, 1);

        __m128d sum01 = _mm_add_pd(h0, h1);
        __m128d sum23 = _mm_add_pd(h2, h3);
        __m128d sum45 = _mm_add_pd(h4, h5);
        __m128d sum67 = _mm_add_pd(h6, h7);

        __m128d sum0123 = _mm_add_pd(sum01, sum23);
        __m128d sum4567 = _mm_add_pd(sum45, sum67);
        __m128d sum_all = _mm_add_pd(sum0123, sum4567);

        __m128d shuf = _mm_shuffle_pd(sum_all, sum_all, 1);
        sum_all = _mm_add_pd(sum_all, shuf);

        double partial[2];
        _mm_storeu_pd(partial, sum_all);
        sum_sq = partial[0];

        // Tail: 8-element blocks
        for (; c + 8 <= cols; c += 8)
        {
            __m256 v = _mm256_loadu_ps(src + c);
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);

            __m256d d0 = _mm256_cvtps_pd(lo);
            __m256d d1 = _mm256_cvtps_pd(hi);

            __m256d sq0 = _mm256_mul_pd(d0, d0);
            __m256d sq1 = _mm256_mul_pd(d1, d1);

            __m128d h0_t = _mm256_castpd256_pd128(sq0);
            __m128d h1_t = _mm256_extractf128_pd(sq0, 1);
            __m128d h2_t = _mm256_castpd256_pd128(sq1);
            __m128d h3_t = _mm256_extractf128_pd(sq1, 1);

            __m128d sum_t = _mm_add_pd(_mm_add_pd(h0_t, h1_t), _mm_add_pd(h2_t, h3_t));
            __m128d shuf_t = _mm_shuffle_pd(sum_t, sum_t, 1);
            sum_t = _mm_add_pd(sum_t, shuf_t);

            double tail_partial[2];
            _mm_storeu_pd(tail_partial, sum_t);
            sum_sq += tail_partial[0];
        }

        // Scalar tail
        for (; c < cols; ++c)
        {
            double val = static_cast<double>(src[c]);
            sum_sq += val * val;
        }

        // Phase 2: Compute inverse RMS
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;
        __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

        // Phase 3: Apply normalization with AVX2
        c = 0;
        for (; c + 8 <= cols; c += 8)
        {
            __m256 src_vec = _mm256_loadu_ps(src + c);
            __m256 gamma_vec = _mm256_loadu_ps(gamma + c);
            __m256 normalized = _mm256_mul_ps(_mm256_mul_ps(src_vec, inv_rms_vec), gamma_vec);
            _mm256_storeu_ps(dst + c, normalized);
        }

        // Scalar tail
        for (; c < cols; ++c)
        {
            dst[c] = src[c] * inv_rms * gamma[c];
        }
    }
#endif

#if defined(__AVX512F__)
    void rmsnorm_fused_row_avx512(
        const float *src,
        const float *gamma,
        float *dst,
        std::size_t cols,
        float epsilon)
    {
        // Phase 1: Compute sum of squares with AVX512
        double sum_sq = 0.0;
        std::size_t c = 0;

        // Multi-accumulator for better ILP
        __m512d acc0 = _mm512_setzero_pd();
        __m512d acc1 = _mm512_setzero_pd();

        for (; c + 32 <= cols; c += 32)
        {
            // Process 32 float values per iteration
            __m512 fp32_0 = _mm512_loadu_ps(src + c);
            __m512 fp32_1 = _mm512_loadu_ps(src + c + 16);

            // Convert to double and accumulate squares
            __m256 lo0 = _mm512_castps512_ps256(fp32_0);
            __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_0), 1));
            __m256 lo1 = _mm512_castps512_ps256(fp32_1);
            __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_1), 1));

            __m512d d0a = _mm512_cvtps_pd(lo0);
            __m512d d0b = _mm512_cvtps_pd(hi0);
            __m512d d1a = _mm512_cvtps_pd(lo1);
            __m512d d1b = _mm512_cvtps_pd(hi1);

            // FMA: acc += d * d
            acc0 = _mm512_fmadd_pd(d0a, d0a, acc0);
            acc0 = _mm512_fmadd_pd(d0b, d0b, acc0);
            acc1 = _mm512_fmadd_pd(d1a, d1a, acc1);
            acc1 = _mm512_fmadd_pd(d1b, d1b, acc1);
        }

        sum_sq += _mm512_reduce_add_pd(acc0);
        sum_sq += _mm512_reduce_add_pd(acc1);

        // Scalar tail
        for (; c < cols; ++c)
        {
            float val = src[c];
            sum_sq += static_cast<double>(val) * static_cast<double>(val);
        }

        // Phase 2: Compute inverse RMS
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

        // Phase 3: Normalize and convert back with AVX512
        c = 0;
        for (; c + 16 <= cols; c += 16)
        {
            __m512 fp32_vals = _mm512_loadu_ps(src + c);
            __m512 gamma_vec = _mm512_loadu_ps(gamma + c);
            __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
            _mm512_storeu_ps(dst + c, normalized);
        }

        // Scalar tail
        for (; c < cols; ++c)
        {
            float val = src[c];
            float normalized = val * inv_rms * gamma[c];
            dst[c] = normalized;
        }
    }
#endif

    // ========================================================================
    // BF16 RMSNorm Implementation (Scalar/AVX2/AVX512)
    // ========================================================================

    // ========== BF16 Conversion Helpers (Moved to top) ==========

    // (Moved to top)

    namespace
    {

        // ========== BF16 RMSNorm Kernels ==========

        /// BF16 RMSNorm row kernel - Scalar path
        __attribute__((always_inline)) inline void bf16_rmsnorm_row_scalar(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            // Phase 1: Compute sum of squares
            double sum_sq = 0.0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                float val = bf16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
            float inv_rms = 1.0f / rms;

            // Phase 3: Normalize and convert back
            for (std::size_t c = 0; c < cols; ++c)
            {
                float val = bf16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * gamma[c];
                dst_row[c] = fp32_to_bf16_scalar(normalized);
            }
        }

#if defined(__AVX512F__)
        /// BF16 RMSNorm row kernel - AVX512 path
        __attribute__((always_inline)) inline void bf16_rmsnorm_row_avx512(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            // Phase 1: Compute sum of squares with AVX512
            double sum_sq = 0.0;
            std::size_t c = 0;

            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();

            for (; c + 32 <= cols; c += 32)
            {
                // Process 32 BF16 values per iteration
                __m512 fp32_0 = bf16_to_fp32_avx512(src_row + c);
                __m512 fp32_1 = bf16_to_fp32_avx512(src_row + c + 16);

                // Convert to double and accumulate squares
                __m256 lo0 = _mm512_castps512_ps256(fp32_0);
                __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_0), 1));
                __m256 lo1 = _mm512_castps512_ps256(fp32_1);
                __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_1), 1));

                __m512d d0a = _mm512_cvtps_pd(lo0);
                __m512d d0b = _mm512_cvtps_pd(hi0);
                __m512d d1a = _mm512_cvtps_pd(lo1);
                __m512d d1b = _mm512_cvtps_pd(hi1);

                acc0 = _mm512_fmadd_pd(d0a, d0a, acc0);
                acc0 = _mm512_fmadd_pd(d0b, d0b, acc0);
                acc1 = _mm512_fmadd_pd(d1a, d1a, acc1);
                acc1 = _mm512_fmadd_pd(d1b, d1b, acc1);
            }

            sum_sq += _mm512_reduce_add_pd(acc0);
            sum_sq += _mm512_reduce_add_pd(acc1);

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = bf16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
            float inv_rms = 1.0f / rms;
            __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

            // Phase 3: Normalize and convert back with AVX512
            c = 0;
            for (; c + 16 <= cols; c += 16)
            {
                __m512 fp32_vals = bf16_to_fp32_avx512(src_row + c);
                __m512 gamma_vec = _mm512_loadu_ps(gamma + c);
                __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
                fp32_to_bf16_avx512(normalized, dst_row + c);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = bf16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * gamma[c];
                dst_row[c] = fp32_to_bf16_scalar(normalized);
            }
        }
#endif

        // ========== FP16 RMSNorm Kernels ==========

        /// FP16 RMSNorm row kernel - Scalar path
        __attribute__((always_inline)) inline void fp16_rmsnorm_row_scalar(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            // Phase 1: Compute sum of squares
            double sum_sq = 0.0;
            for (std::size_t c = 0; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
            float inv_rms = 1.0f / rms;

            // Phase 3: Normalize and convert back
            for (std::size_t c = 0; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * gamma[c];
                dst_row[c] = fp32_to_fp16_scalar(normalized);
            }
        }

#if defined(__AVX512F__)
        /// FP16 RMSNorm row kernel - AVX512 path
        __attribute__((always_inline)) inline void fp16_rmsnorm_row_avx512(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            // Phase 1: Compute sum of squares with AVX512
            double sum_sq = 0.0;
            std::size_t c = 0;

            __m512d acc0 = _mm512_setzero_pd();
            __m512d acc1 = _mm512_setzero_pd();

            for (; c + 32 <= cols; c += 32)
            {
                // Process 32 FP16 values per iteration
                __m512 fp32_0 = fp16_to_fp32_avx512(src_row + c);
                __m512 fp32_1 = fp16_to_fp32_avx512(src_row + c + 16);

                // Convert to double and accumulate squares
                __m256 lo0 = _mm512_castps512_ps256(fp32_0);
                __m256 hi0 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_0), 1));
                __m256 lo1 = _mm512_castps512_ps256(fp32_1);
                __m256 hi1 = _mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(fp32_1), 1));

                __m512d d0a = _mm512_cvtps_pd(lo0);
                __m512d d0b = _mm512_cvtps_pd(hi0);
                __m512d d1a = _mm512_cvtps_pd(lo1);
                __m512d d1b = _mm512_cvtps_pd(hi1);

                acc0 = _mm512_fmadd_pd(d0a, d0a, acc0);
                acc0 = _mm512_fmadd_pd(d0b, d0b, acc0);
                acc1 = _mm512_fmadd_pd(d1a, d1a, acc1);
                acc1 = _mm512_fmadd_pd(d1b, d1b, acc1);
            }

            sum_sq += _mm512_reduce_add_pd(acc0);
            sum_sq += _mm512_reduce_add_pd(acc1);

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                sum_sq += static_cast<double>(val) * static_cast<double>(val);
            }

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
            float inv_rms = 1.0f / rms;
            __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

            // Phase 3: Normalize and convert back with AVX512
            c = 0;
            for (; c + 16 <= cols; c += 16)
            {
                __m512 fp32_vals = fp16_to_fp32_avx512(src_row + c);
                __m512 gamma_vec = _mm512_loadu_ps(gamma + c);
                __m512 normalized = _mm512_mul_ps(_mm512_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
                fp32_to_fp16_avx512(normalized, dst_row + c);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * gamma[c];
                dst_row[c] = fp32_to_fp16_scalar(normalized);
            }
        }
#endif

#if defined(__AVX2__) && defined(__F16C__)
        /// FP16 RMSNorm row kernel - AVX2+F16C path
        __attribute__((always_inline)) inline void fp16_rmsnorm_row_avx2(
            const uint16_t *src_row,
            const float *gamma,
            uint16_t *dst_row,
            std::size_t cols,
            float epsilon)
        {
            // Phase 1: Compute sum of squares with AVX2
            float sum_sq_f = 0.0f;
            std::size_t c = 0;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();

            for (; c + 16 <= cols; c += 16)
            {
                __m256 fp32_0 = fp16_to_fp32_avx2(src_row + c);
                __m256 fp32_1 = fp16_to_fp32_avx2(src_row + c + 8);

                acc0 = _mm256_fmadd_ps(fp32_0, fp32_0, acc0);
                acc1 = _mm256_fmadd_ps(fp32_1, fp32_1, acc1);
            }

            // Horizontal reduction
            __m256 acc_combined = _mm256_add_ps(acc0, acc1);
            __m128 lo = _mm256_castps256_ps128(acc_combined);
            __m128 hi = _mm256_extractf128_ps(acc_combined, 1);
            __m128 sum4 = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(sum4);
            __m128 sums = _mm_add_ps(sum4, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            _mm_store_ss(&sum_sq_f, sums);

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                sum_sq_f += val * val;
            }

            // Phase 2: Compute inverse RMS
            float rms = std::sqrt(sum_sq_f / cols + epsilon);
            float inv_rms = 1.0f / rms;
            __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

            // Phase 3: Normalize and convert back
            c = 0;
            for (; c + 8 <= cols; c += 8)
            {
                __m256 fp32_vals = fp16_to_fp32_avx2(src_row + c);
                __m256 gamma_vec = _mm256_loadu_ps(gamma + c);
                __m256 normalized = _mm256_mul_ps(_mm256_mul_ps(fp32_vals, inv_rms_vec), gamma_vec);
                fp32_to_fp16_avx2(normalized, dst_row + c);
            }

            // Scalar tail
            for (; c < cols; ++c)
            {
                float val = fp16_to_fp32_scalar(src_row[c]);
                float normalized = val * inv_rms * gamma[c];
                dst_row[c] = fp32_to_fp16_scalar(normalized);
            }
        }
#endif

    } // anonymous namespace

    // ========================================================================
    // Public Implementations
    // ========================================================================

    void rmsnorm_fused_bf16_vectorized(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        bool parallel = want_parallel(rows, cols, opts);

        auto do_row_work = [&](std::size_t r)
        {
            const uint16_t *src_row = src + r * cols;
            uint16_t *dst_row = dst + r * cols;

#if defined(__AVX512F__)
            if (!opts.force_scalar)
            {
                bf16_rmsnorm_row_avx512(src_row, gamma, dst_row, cols, epsilon);
            }
            else
#endif
            {
                bf16_rmsnorm_row_scalar(src_row, gamma, dst_row, cols, epsilon);
            }
        };

        if (parallel)
        {
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (std::size_t r = 0; r < rows; ++r)
                {
                    do_row_work(r);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            for (std::size_t r = 0; r < rows; ++r)
            {
                do_row_work(r);
            }
        }
    }

    void rmsnorm_fused_fp16_vectorized(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        bool parallel = want_parallel(rows, cols, opts);

        auto do_row_work = [&](std::size_t r)
        {
            const uint16_t *src_row = src + r * cols;
            uint16_t *dst_row = dst + r * cols;

#if defined(__AVX512F__)
            if (!opts.force_scalar)
            {
                fp16_rmsnorm_row_avx512(src_row, gamma, dst_row, cols, epsilon);
            }
            else
#elif defined(__AVX2__) && defined(__F16C__)
            if (!opts.force_scalar)
            {
                fp16_rmsnorm_row_avx2(src_row, gamma, dst_row, cols, epsilon);
            }
            else
#endif
            {
                fp16_rmsnorm_row_scalar(src_row, gamma, dst_row, cols, epsilon);
            }
        };

        if (parallel)
        {
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (std::size_t r = 0; r < rows; ++r)
                {
                    do_row_work(r);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
        }
        else
        {
            for (std::size_t r = 0; r < rows; ++r)
            {
                do_row_work(r);
            }
        }
    }

    void rmsnorm_quantize_row_scalar(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
        // 1. Compute RMS
        double sum_sq = 0.0;
        for (int i = 0; i < cols; ++i)
        {
            sum_sq += input_row[i] * input_row[i];
        }
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;

        // 2. Find max abs of normalized values
        float max_abs = 0.0f;
        for (int i = 0; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            max_abs = std::max(max_abs, std::abs(val));
        }

        // 3. Compute scale and quantize
        float scale = max_abs / 127.0f;
        if (scale < 1e-8f)
            scale = 1e-8f;
        out_scale = scale;
        float inv_scale = 1.0f / scale;

        for (int i = 0; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            int32_t q = std::round(val * inv_scale);
            if (q > 127)
                q = 127;
            if (q < -127)
                q = -127;
            output_row[i] = static_cast<int8_t>(q);
        }
    }

    void rmsnorm_quantize_row_avx2(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
#if defined(__AVX2__)
        // 1. Compute RMS
        __m256 sum_sq_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i <= cols - 8; i += 8)
        {
            __m256 v = _mm256_loadu_ps(input_row + i);
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
        }
        // Reduce sum_sq_vec
        __m128 lo = _mm256_castps256_ps128(sum_sq_vec);
        __m128 hi = _mm256_extractf128_ps(sum_sq_vec, 1);
        __m128 sum4 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum4);
        __m128 sums = _mm_add_ps(sum4, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        float sum_sq = _mm_cvtss_f32(sums);

        for (; i < cols; ++i)
        {
            sum_sq += input_row[i] * input_row[i];
        }

        float rms = std::sqrt(sum_sq / cols + epsilon);
        float inv_rms = 1.0f / rms;
        __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

        // 2. Find max abs
        __m256 max_abs_vec = _mm256_setzero_ps();
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        i = 0;
        for (; i <= cols - 8; i += 8)
        {
            __m256 v = _mm256_loadu_ps(input_row + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 norm = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_vec), w);
            __m256 abs_norm = _mm256_and_ps(norm, abs_mask);
            max_abs_vec = _mm256_max_ps(max_abs_vec, abs_norm);
        }
        // Reduce max_abs_vec
        lo = _mm256_castps256_ps128(max_abs_vec);
        hi = _mm256_extractf128_ps(max_abs_vec, 1);
        __m128 max4 = _mm_max_ps(lo, hi);
        shuf = _mm_movehdup_ps(max4);
        __m128 maxs = _mm_max_ps(max4, shuf);
        shuf = _mm_movehl_ps(shuf, maxs);
        maxs = _mm_max_ss(maxs, shuf);
        float max_abs = _mm_cvtss_f32(maxs);

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            max_abs = std::max(max_abs, std::abs(val));
        }

        // 3. Compute scale and quantize
        float scale = max_abs / 127.0f;
        if (scale < 1e-8f)
            scale = 1e-8f;
        out_scale = scale;
        float inv_scale = 1.0f / scale;
        __m256 inv_scale_vec = _mm256_set1_ps(inv_scale);

        i = 0;
        for (; i <= cols - 8; i += 8)
        {
            __m256 v = _mm256_loadu_ps(input_row + i);
            __m256 w = _mm256_loadu_ps(weight + i);
            __m256 norm = _mm256_mul_ps(_mm256_mul_ps(v, inv_rms_vec), w);
            __m256 scaled = _mm256_mul_ps(norm, inv_scale_vec);
            __m256i q = _mm256_cvtps_epi32(scaled); // Round to nearest integer

            int32_t tmp[8];
            _mm256_storeu_si256((__m256i *)tmp, q);
            for (int k = 0; k < 8; ++k)
            {
                int32_t val = tmp[k];
                if (val > 127)
                    val = 127;
                if (val < -127)
                    val = -127;
                output_row[i + k] = (int8_t)val;
            }
        }

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            int32_t q = std::round(val * inv_scale);
            if (q > 127)
                q = 127;
            if (q < -127)
                q = -127;
            output_row[i] = static_cast<int8_t>(q);
        }
#else
        rmsnorm_quantize_row_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
#endif
    }

    void rmsnorm_quantize_row_avx512(
        const float *input_row,
        const float *weight,
        int8_t *output_row,
        float &out_scale,
        int cols,
        float epsilon)
    {
#if defined(__AVX512F__)
        // 1. Compute RMS
        __m512 sum_sq_vec = _mm512_setzero_ps();
        int i = 0;
        for (; i <= cols - 16; i += 16)
        {
            __m512 v = _mm512_loadu_ps(input_row + i);
            sum_sq_vec = _mm512_fmadd_ps(v, v, sum_sq_vec);
        }
        float sum_sq = _mm512_reduce_add_ps(sum_sq_vec);

        for (; i < cols; ++i)
        {
            sum_sq += input_row[i] * input_row[i];
        }

        float rms = std::sqrt(sum_sq / cols + epsilon);
        float inv_rms = 1.0f / rms;
        __m512 inv_rms_vec = _mm512_set1_ps(inv_rms);

        // 2. Find max abs
        __m512 max_abs_vec = _mm512_setzero_ps();
        __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));

        i = 0;
        for (; i <= cols - 16; i += 16)
        {
            __m512 v = _mm512_loadu_ps(input_row + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 norm = _mm512_mul_ps(_mm512_mul_ps(v, inv_rms_vec), w);
            __m512 abs_norm = _mm512_and_ps(norm, abs_mask);
            max_abs_vec = _mm512_max_ps(max_abs_vec, abs_norm);
        }
        float max_abs = _mm512_reduce_max_ps(max_abs_vec);

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            max_abs = std::max(max_abs, std::abs(val));
        }

        // 3. Compute scale and quantize
        float scale = max_abs / 127.0f;
        if (scale < 1e-8f)
            scale = 1e-8f;
        out_scale = scale;
        float inv_scale = 1.0f / scale;
        __m512 inv_scale_vec = _mm512_set1_ps(inv_scale);

        i = 0;
        for (; i <= cols - 16; i += 16)
        {
            __m512 v = _mm512_loadu_ps(input_row + i);
            __m512 w = _mm512_loadu_ps(weight + i);
            __m512 norm = _mm512_mul_ps(_mm512_mul_ps(v, inv_rms_vec), w);
            __m512 scaled = _mm512_mul_ps(norm, inv_scale_vec);
            __m512i q = _mm512_cvtps_epi32(scaled);

#if defined(__AVX512BW__)
            __m128i q8 = _mm512_cvtepi32_epi8(q);
            _mm_storeu_si128((__m128i *)(output_row + i), q8);
#else
            int32_t tmp[16];
            _mm512_storeu_si512(tmp, q);
            for (int k = 0; k < 16; ++k)
            {
                int32_t val = tmp[k];
                if (val > 127)
                    val = 127;
                if (val < -127)
                    val = -127;
                output_row[i + k] = (int8_t)val;
            }
#endif
        }

        for (; i < cols; ++i)
        {
            float val = input_row[i] * inv_rms * weight[i];
            int32_t q = std::round(val * inv_scale);
            if (q > 127)
                q = 127;
            if (q < -127)
                q = -127;
            output_row[i] = static_cast<int8_t>(q);
        }
#else
        rmsnorm_quantize_row_scalar(input_row, weight, output_row, out_scale, cols, epsilon);
#endif
    }

    // Implement per-row primitives (wrappers)
    void rmsnorm_fused_row_bf16_scalar(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
        bf16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
    }

    void rmsnorm_fused_row_bf16_avx2(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
        // Fallback to scalar if no AVX2 BF16 implementation
        bf16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
    }

    void rmsnorm_fused_row_bf16_avx512(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
#if defined(__AVX512F__)
        bf16_rmsnorm_row_avx512(src, gamma, dst, cols, epsilon);
#else
        bf16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
#endif
    }

    void rmsnorm_fused_row_fp16_scalar(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
        fp16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
    }

    void rmsnorm_fused_row_fp16_avx2(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
#if defined(__AVX2__) && defined(__F16C__)
        fp16_rmsnorm_row_avx2(src, gamma, dst, cols, epsilon);
#else
        fp16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
#endif
    }

    void rmsnorm_fused_row_fp16_avx512(
        const uint16_t *src,
        const float *gamma,
        uint16_t *dst,
        std::size_t cols,
        float epsilon)
    {
#if defined(__AVX512F__)
        fp16_rmsnorm_row_avx512(src, gamma, dst, cols, epsilon);
#else
        fp16_rmsnorm_row_scalar(src, gamma, dst, cols, epsilon);
#endif
    }

    // ========================================================================
    // Q8_1 Integer-Space RMSNorm Implementation
    // ========================================================================

    namespace
    {
        /**
         * @brief FP16 to FP32 conversion (for Q8_1 scale factors)
         */
        __attribute__((always_inline)) inline float fp16_to_fp32_q8(uint16_t h)
        {
            // Use hardware instruction if available
#if defined(__F16C__)
            return _cvtsh_ss(h);
#else
            // Software fallback
            uint32_t sign = (h & 0x8000U) << 16;
            uint32_t exp = (h & 0x7C00U) >> 10;
            uint32_t mantissa = h & 0x03FFU;
            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mantissa == 0)
                    fp32_bits = sign;
                else
                {
                    exp = 1;
                    while ((mantissa & 0x0400U) == 0)
                    {
                        mantissa <<= 1;
                        exp--;
                    }
                    mantissa &= 0x03FFU;
                    fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
                }
            }
            else if (exp == 0x1F)
            {
                fp32_bits = sign | 0x7F800000U | (mantissa << 13);
            }
            else
            {
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mantissa << 13);
            }
            float result;
            std::memcpy(&result, &fp32_bits, sizeof(float));
            return result;
#endif
        }

        /**
         * @brief FP32 to FP16 conversion (for Q8_1 scale factors)
         */
        __attribute__((always_inline)) inline uint16_t fp32_to_fp16_q8(float f)
        {
#if defined(__F16C__)
            return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT);
#else
            uint32_t bits;
            std::memcpy(&bits, &f, sizeof(float));
            uint32_t sign = (bits >> 16) & 0x8000U;
            int32_t exp = ((bits >> 23) & 0xFFU) - 127 + 15;
            uint32_t mantissa = (bits >> 13) & 0x03FFU;
            if (exp <= 0)
            {
                if (exp < -10)
                    return sign;
                mantissa |= 0x0400U;
                mantissa >>= (1 - exp);
                return sign | mantissa;
            }
            else if (exp >= 0x1F)
            {
                return sign | 0x7C00U;
            }
            else
            {
                return sign | (exp << 10) | mantissa;
            }
#endif
        }

#if defined(__AVX512F__)
        /**
         * @brief Compute sum of squares for 32 int8 values using AVX512
         *
         * Uses VPMADDUBSW + VPMADDWD for efficient int8*int8 -> int32 accumulation.
         * Note: We treat qs as signed int8, so we use a different approach than VNNI.
         */
        __attribute__((always_inline)) inline int32_t compute_int8_sumsq_avx512(const int8_t *qs)
        {
            // Load 32 int8 values
            __m256i v8 = _mm256_loadu_si256((const __m256i *)qs);

            // Convert to int16 (need signed extension)
            __m512i v16 = _mm512_cvtepi8_epi16(v8);

            // Square: v16 * v16 -> int32 (using PMADDWD with self)
            // PMADDWD: (a0*b0 + a1*b1), (a2*b2 + a3*b3), ...
            __m512i v32 = _mm512_madd_epi16(v16, v16);

            // Horizontal sum
            return _mm512_reduce_add_epi32(v32);
        }
#endif

#if defined(__AVX2__)
        /**
         * @brief Compute sum of squares for 32 int8 values using AVX2
         */
        __attribute__((always_inline)) inline int32_t compute_int8_sumsq_avx2(const int8_t *qs)
        {
            // Load 32 int8 values as two 128-bit chunks
            __m128i v8_lo = _mm_loadu_si128((const __m128i *)qs);
            __m128i v8_hi = _mm_loadu_si128((const __m128i *)(qs + 16));

            // Convert to int16
            __m256i v16_lo = _mm256_cvtepi8_epi16(v8_lo);
            __m256i v16_hi = _mm256_cvtepi8_epi16(v8_hi);

            // Square using PMADDWD
            __m256i v32_lo = _mm256_madd_epi16(v16_lo, v16_lo);
            __m256i v32_hi = _mm256_madd_epi16(v16_hi, v16_hi);

            // Add the two halves
            __m256i v32_sum = _mm256_add_epi32(v32_lo, v32_hi);

            // Horizontal sum (AVX2 doesn't have _mm256_reduce_add_epi32)
            __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(v32_sum),
                                           _mm256_extracti128_si256(v32_sum, 1));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(1, 0, 3, 2)));
            sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_SHUFFLE(2, 3, 0, 1)));
            return _mm_cvtsi128_si32(sum128);
        }
#endif

        /**
         * @brief Compute sum of squares for 32 int8 values (scalar fallback)
         */
        __attribute__((always_inline)) inline int32_t compute_int8_sumsq_scalar(const int8_t *qs)
        {
            int32_t sum = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t val = static_cast<int32_t>(qs[i]);
                sum += val * val;
            }
            return sum;
        }

        /**
         * @brief Dispatch to best available SIMD for int8 sum-of-squares
         */
        __attribute__((always_inline)) inline int32_t compute_int8_sumsq(const int8_t *qs)
        {
#if defined(__AVX512F__)
            return compute_int8_sumsq_avx512(qs);
#elif defined(__AVX2__)
            return compute_int8_sumsq_avx2(qs);
#else
            return compute_int8_sumsq_scalar(qs);
#endif
        }

    } // anonymous namespace

    // =========================================================================
    // PURE INTEGER RMSNorm Implementation
    // =========================================================================
    //
    // This implementation performs RMSNorm entirely in integer domain by:
    // 1. Computing sum(qs²) in pure integer (no FP16 scale involved yet)
    // 2. Using a simplified model: assume all blocks have similar scale
    //    OR use a "representative scale" approach
    // 3. Integer inverse square root via Newton-Raphson
    // 4. Pre-quantized gamma weights (int16)
    //
    // KEY SIMPLIFICATION: For activations, blocks within a row typically have
    // similar scales (within 2-4x). We use the geometric mean of scales.
    // =========================================================================

    namespace
    {
        /**
         * @brief Fast integer square root using binary search
         * Returns floor(sqrt(x))
         */
        __attribute__((always_inline)) inline uint32_t isqrt_fast(uint64_t x)
        {
            if (x == 0)
                return 0;
            if (x == 1)
                return 1;

            // Initial estimate using CLZ
            int bits = 64 - __builtin_clzll(x);
            uint64_t result = 1ULL << ((bits + 1) / 2);

            // Newton-Raphson: 3-4 iterations suffice for 64-bit
            for (int i = 0; i < 4; ++i)
            {
                result = (result + x / result) >> 1;
            }

            // Adjust if we overshot
            while (result * result > x)
                --result;

            return static_cast<uint32_t>(result);
        }

        /**
         * @brief Compute integer inverse sqrt in Q16 format
         * Input: x (raw integer)
         * Output: (1/sqrt(x)) << 16, or 0 if x is 0
         */
        __attribute__((always_inline)) inline uint32_t inv_sqrt_q16(uint64_t x)
        {
            if (x == 0)
                return 0xFFFF; // Large value for zero

            uint32_t sqrt_x = isqrt_fast(x);
            if (sqrt_x == 0)
                return 0xFFFF;

            // inv_sqrt = (1 << 16) / sqrt_x, but we want more precision
            // Use (1 << 32) / sqrt_x, then adjust
            return static_cast<uint32_t>((1ULL << 24) / sqrt_x); // Q24 format
        }

#if defined(__AVX512F__)
        /**
         * @brief AVX512 vectorized sum of squares for 32 int8 values
         */
        __attribute__((always_inline)) inline int32_t compute_sumsq_avx512(const int8_t *qs)
        {
            __m256i v8 = _mm256_loadu_si256((const __m256i *)qs);
            __m512i v16 = _mm512_cvtepi8_epi16(v8);
            __m512i v32 = _mm512_madd_epi16(v16, v16);
            return _mm512_reduce_add_epi32(v32);
        }
#endif
    }

    /**
     * @brief Unrolled (2x) Single-Pass RMSNorm for Q8_1
     *
     * Optimizes the single-pass approach by processing 2 blocks per iteration.
     * This hides the latency of the max_abs -> scale dependency chain by
     * interleaving instructions for two independent blocks.
     *
     * Maintains strict per-block correctness (unlike the global-scale two-pass approach).
     */
    void rmsnorm_q8_1_pure_integer_row_unrolled(
        const Q8_1Block *input,
        const int16_t *gamma_q12,
        Q8_1Block *output,
        std::size_t blocks_per_row,
        int32_t epsilon_scaled)
    {
        if (!input || !gamma_q12 || !output || blocks_per_row == 0)
            return;

        const size_t cols = blocks_per_row * 32;

        // ====================================================================
        // Phase 1 & 2: Compute Sum(d² * sum_qs²) using vector accumulation
        // ====================================================================

#if defined(__AVX512F__)
        __m512 v_total_sum = _mm512_setzero_ps();

        // Unrolled sum loop (4x)
        size_t b = 0;
        for (; b + 4 <= blocks_per_row; b += 4)
        {
            // Block 0
            float d0 = fp16_to_fp32_q8(input[b].d);
            __m512 v_d0_sq = _mm512_set1_ps(d0 * d0);
            __m256i qs8_0 = _mm256_loadu_si256((const __m256i *)input[b].qs);
            __m512i qs16_0 = _mm512_cvtepi8_epi16(qs8_0);
            __m512i qs_sq0 = _mm512_madd_epi16(qs16_0, qs16_0);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq0), v_d0_sq, v_total_sum);

            // Block 1
            float d1 = fp16_to_fp32_q8(input[b + 1].d);
            __m512 v_d1_sq = _mm512_set1_ps(d1 * d1);
            __m256i qs8_1 = _mm256_loadu_si256((const __m256i *)input[b + 1].qs);
            __m512i qs16_1 = _mm512_cvtepi8_epi16(qs8_1);
            __m512i qs_sq1 = _mm512_madd_epi16(qs16_1, qs16_1);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq1), v_d1_sq, v_total_sum);

            // Block 2
            float d2 = fp16_to_fp32_q8(input[b + 2].d);
            __m512 v_d2_sq = _mm512_set1_ps(d2 * d2);
            __m256i qs8_2 = _mm256_loadu_si256((const __m256i *)input[b + 2].qs);
            __m512i qs16_2 = _mm512_cvtepi8_epi16(qs8_2);
            __m512i qs_sq2 = _mm512_madd_epi16(qs16_2, qs16_2);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq2), v_d2_sq, v_total_sum);

            // Block 3
            float d3 = fp16_to_fp32_q8(input[b + 3].d);
            __m512 v_d3_sq = _mm512_set1_ps(d3 * d3);
            __m256i qs8_3 = _mm256_loadu_si256((const __m256i *)input[b + 3].qs);
            __m512i qs16_3 = _mm512_cvtepi8_epi16(qs8_3);
            __m512i qs_sq3 = _mm512_madd_epi16(qs16_3, qs16_3);
            v_total_sum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(qs_sq3), v_d3_sq, v_total_sum);
        }

        double total_sum_sq = static_cast<double>(_mm512_reduce_add_ps(v_total_sum));
#else
        double total_sum_sq = 0.0;
        size_t b = 0;
#endif
        for (; b < blocks_per_row; ++b)
        {
            float d = fp16_to_fp32_q8(input[b].d);
            int32_t s = 0;
#if defined(__AVX512F__)
            __m256i qs_8 = _mm256_loadu_si256((const __m256i *)input[b].qs);
            __m512i qs_16 = _mm512_cvtepi8_epi16(qs_8);
            __m512i qs_sq = _mm512_madd_epi16(qs_16, qs_16);
            s = _mm512_reduce_add_epi32(qs_sq);
#else
            for (int i = 0; i < 32; ++i)
                s += input[b].qs[i] * input[b].qs[i];
#endif
            total_sum_sq += static_cast<double>(d) * d * s;
        }

        // ====================================================================
        // Phase 3: Compute RMS and inverse RMS
        // ====================================================================
        double mean_sq = total_sum_sq / cols;
        mean_sq += 1e-5; // Epsilon
        double rms = std::sqrt(mean_sq);
        double inv_rms_f = 16777216.0 / rms;
        if (inv_rms_f > 2147483647.0)
            inv_rms_f = 2147483647.0;
        uint32_t inv_rms_q24 = static_cast<uint32_t>(inv_rms_f);
        if (inv_rms_q24 == 0)
            inv_rms_q24 = 1;

        // ====================================================================
        // Phase 4: Apply normalization (Unrolled 2x)
        // ====================================================================

#if defined(__AVX512F__)
        // Optimized implementation using float scaling to avoid 64-bit integer ops
        float inv_rms_float = 1.0f / static_cast<float>(rms);
        __m512 v_inv_rms_f = _mm512_set1_ps(inv_rms_float);
        __m512i v_127_i = _mm512_set1_epi32(127);
        __m512i v_neg127_i = _mm512_set1_epi32(-127);
        __m512i v_abs_mask = _mm512_set1_epi32(0x7FFFFFFF);

        // Unrolled loop (4x) with Vectorized Division
        b = 0;
        for (; b + 4 <= blocks_per_row; b += 4)
        {
            // Block 0
            const Q8_1Block &in0 = input[b];
            __m256i qs8_0 = _mm256_loadu_si256((const __m256i *)in0.qs);
            __m512i qs16_0 = _mm512_cvtepi8_epi16(qs8_0);
            __m512i g16_0 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + b * 32));

            __m512i qs32_lo_0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_0));
            __m512i qs32_hi_0 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_0, 1));
            __m512i g32_lo_0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_0));
            __m512i g32_hi_0 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_0, 1));

            __m512i prod32_lo_0 = _mm512_mullo_epi32(qs32_lo_0, g32_lo_0);
            __m512i prod32_hi_0 = _mm512_mullo_epi32(qs32_hi_0, g32_hi_0);

            __m512i max_int_v_0 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_0), _mm512_abs_epi32(prod32_hi_0));
            float m0 = (float)_mm512_reduce_max_epi32(max_int_v_0);

            // Block 1
            const Q8_1Block &in1 = input[b + 1];
            __m256i qs8_1 = _mm256_loadu_si256((const __m256i *)in1.qs);
            __m512i qs16_1 = _mm512_cvtepi8_epi16(qs8_1);
            __m512i g16_1 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + (b + 1) * 32));

            __m512i qs32_lo_1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_1));
            __m512i qs32_hi_1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_1, 1));
            __m512i g32_lo_1 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_1));
            __m512i g32_hi_1 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_1, 1));

            __m512i prod32_lo_1 = _mm512_mullo_epi32(qs32_lo_1, g32_lo_1);
            __m512i prod32_hi_1 = _mm512_mullo_epi32(qs32_hi_1, g32_hi_1);

            __m512i max_int_v_1 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_1), _mm512_abs_epi32(prod32_hi_1));
            float m1 = (float)_mm512_reduce_max_epi32(max_int_v_1);

            // Block 2
            const Q8_1Block &in2 = input[b + 2];
            __m256i qs8_2 = _mm256_loadu_si256((const __m256i *)in2.qs);
            __m512i qs16_2 = _mm512_cvtepi8_epi16(qs8_2);
            __m512i g16_2 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + (b + 2) * 32));

            __m512i qs32_lo_2 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_2));
            __m512i qs32_hi_2 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_2, 1));
            __m512i g32_lo_2 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_2));
            __m512i g32_hi_2 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_2, 1));

            __m512i prod32_lo_2 = _mm512_mullo_epi32(qs32_lo_2, g32_lo_2);
            __m512i prod32_hi_2 = _mm512_mullo_epi32(qs32_hi_2, g32_hi_2);

            __m512i max_int_v_2 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_2), _mm512_abs_epi32(prod32_hi_2));
            float m2 = (float)_mm512_reduce_max_epi32(max_int_v_2);

            // Block 3
            const Q8_1Block &in3 = input[b + 3];
            __m256i qs8_3 = _mm256_loadu_si256((const __m256i *)in3.qs);
            __m512i qs16_3 = _mm512_cvtepi8_epi16(qs8_3);
            __m512i g16_3 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + (b + 3) * 32));

            __m512i qs32_lo_3 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16_3));
            __m512i qs32_hi_3 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16_3, 1));
            __m512i g32_lo_3 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16_3));
            __m512i g32_hi_3 = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16_3, 1));

            __m512i prod32_lo_3 = _mm512_mullo_epi32(qs32_lo_3, g32_lo_3);
            __m512i prod32_hi_3 = _mm512_mullo_epi32(qs32_hi_3, g32_hi_3);

            __m512i max_int_v_3 = _mm512_max_epi32(_mm512_abs_epi32(prod32_lo_3), _mm512_abs_epi32(prod32_hi_3));
            float m3 = (float)_mm512_reduce_max_epi32(max_int_v_3);

            // Vectorized Reciprocal
            __m128 v_max = _mm_set_ps(m3, m2, m1, m0);
            v_max = _mm_max_ps(v_max, _mm_set1_ps(1e-9f)); // Avoid div/0
            __m128 v_rcp = _mm_rcp_ps(v_max);
            // Newton-Raphson: x1 = x0 * (2 - a * x0)
            v_rcp = _mm_mul_ps(v_rcp, _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(v_max, v_rcp)));
            __m128 v_scale = _mm_mul_ps(v_rcp, _mm_set1_ps(127.0f));

            // Extract scales
            float s0 = _mm_cvtss_f32(v_scale);
            float s1 = _mm_cvtss_f32(_mm_shuffle_ps(v_scale, v_scale, 0x55));
            float s2 = _mm_cvtss_f32(_mm_shuffle_ps(v_scale, v_scale, 0xAA));
            float s3 = _mm_cvtss_f32(_mm_shuffle_ps(v_scale, v_scale, 0xFF));

            // Finish Block 0
            {
                Q8_1Block &out = output[b];
                float d_in = fp16_to_fp32_q8(in0.d);
                float d_out_val = d_in * m0 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s0 = _mm512_set1_ps(s0);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_0);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_0);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s0));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s0));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }

            // Finish Block 1
            {
                Q8_1Block &out = output[b + 1];
                float d_in = fp16_to_fp32_q8(in1.d);
                float d_out_val = d_in * m1 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s1 = _mm512_set1_ps(s1);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_1);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_1);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s1));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s1));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }

            // Finish Block 2
            {
                Q8_1Block &out = output[b + 2];
                float d_in = fp16_to_fp32_q8(in2.d);
                float d_out_val = d_in * m2 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s2 = _mm512_set1_ps(s2);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_2);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_2);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s2));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s2));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }

            // Finish Block 3
            {
                Q8_1Block &out = output[b + 3];
                float d_in = fp16_to_fp32_q8(in3.d);
                float d_out_val = d_in * m3 * inv_rms_float * (1.0f / (127.0f * 2048.0f));
                out.d = fp32_to_fp16_q8(d_out_val);

                __m512 v_s3 = _mm512_set1_ps(s3);
                __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo_3);
                __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi_3);
                __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_s3));
                __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_s3));

                final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
                final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
                final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
                final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

                __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
                __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
                __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
                _mm256_storeu_si256((__m256i *)out.qs, out8);
                out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
            }
        }
        // Remaining blocks
        for (; b < blocks_per_row; ++b)
        {
            const Q8_1Block &in = input[b];
            Q8_1Block &out = output[b];
            float d_in = fp16_to_fp32_q8(in.d);

            __m256i qs8 = _mm256_loadu_si256((const __m256i *)in.qs);
            __m512i qs16 = _mm512_cvtepi8_epi16(qs8);
            __m512i g16 = _mm512_loadu_si512((const __m512i *)(gamma_q12 + b * 32));

            __m512i qs32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(qs16));
            __m512i qs32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(qs16, 1));
            __m512i g32_lo = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g16));
            __m512i g32_hi = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(g16, 1));

            __m512i prod32_lo = _mm512_mullo_epi32(qs32_lo, g32_lo);
            __m512i prod32_hi = _mm512_mullo_epi32(qs32_hi, g32_hi);

            __m512i abs_int_lo = _mm512_abs_epi32(prod32_lo);
            __m512i abs_int_hi = _mm512_abs_epi32(prod32_hi);
            __m512i max_int_v = _mm512_max_epi32(abs_int_lo, abs_int_hi);
            int32_t max_int = _mm512_reduce_max_epi32(max_int_v);
            float max_prod = static_cast<float>(max_int);

            float d_out_val = (max_prod > 1e-9f) ? (d_in * max_prod * inv_rms_float / (127.0f * 2048.0f)) : 0.0f;
            out.d = fp32_to_fp16_q8(d_out_val);

            float combined_scale_val = (max_prod > 1e-9f) ? (127.0f / max_prod) : 0.0f;
            __m512 v_combined_scale = _mm512_set1_ps(combined_scale_val);

            __m512 prod_ps_lo = _mm512_cvtepi32_ps(prod32_lo);
            __m512 prod_ps_hi = _mm512_cvtepi32_ps(prod32_hi);
            __m512i final_i_lo = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_lo, v_combined_scale));
            __m512i final_i_hi = _mm512_cvtps_epi32(_mm512_mul_ps(prod_ps_hi, v_combined_scale));

            final_i_lo = _mm512_max_epi32(final_i_lo, v_neg127_i);
            final_i_lo = _mm512_min_epi32(final_i_lo, v_127_i);
            final_i_hi = _mm512_max_epi32(final_i_hi, v_neg127_i);
            final_i_hi = _mm512_min_epi32(final_i_hi, v_127_i);

            __m128i out8_lo = _mm512_cvtepi32_epi8(final_i_lo);
            __m128i out8_hi = _mm512_cvtepi32_epi8(final_i_hi);
            __m256i out8 = _mm256_inserti128_si256(_mm256_castsi128_si256(out8_lo), out8_hi, 1);
            _mm256_storeu_si256((__m256i *)out.qs, out8);
            out.sum_qs = static_cast<int16_t>(_mm512_reduce_add_epi32(final_i_lo) + _mm512_reduce_add_epi32(final_i_hi));
        }
#endif
        // Scalar fallback for remaining blocks
        for (; b < blocks_per_row; ++b)
        {
            const Q8_1Block &in_blk = input[b];
            Q8_1Block &out_blk = output[b];
            float d_in = fp16_to_fp32_q8(in_blk.d);

            int32_t y_raw[32];
            int32_t max_abs = 0;
            for (int i = 0; i < 32; ++i)
            {
                size_t ch = b * 32 + i;
                int64_t term = static_cast<int64_t>(in_blk.qs[i]) * gamma_q12[ch];
                int64_t res = term * inv_rms_q24;
                y_raw[i] = static_cast<int32_t>(res >> 20);
                max_abs = std::max(max_abs, std::abs(y_raw[i]));
            }

            float d_out;
            float scale_factor = 1.0f;
            if (max_abs == 0)
            {
                d_out = 0.0f;
                scale_factor = 0.0f;
            }
            else
            {
                d_out = d_in * static_cast<float>(max_abs) / (127.0f * 32768.0f);
                scale_factor = 127.0f / static_cast<float>(max_abs);
            }

            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t y = static_cast<int32_t>(std::round(y_raw[i] * scale_factor));
                y = (y > 127) ? 127 : (y < -127) ? -127
                                                 : y;
                out_blk.qs[i] = static_cast<int8_t>(y);
                sum_qs += y;
            }
            out_blk.sum_qs = static_cast<int16_t>(sum_qs);
            out_blk.d = fp32_to_fp16_q8(d_out);
        }
    }

    void rmsnorm_q8_1_pure_integer_row(
        const Q8_1Block *input,
        const int16_t *gamma_q12, // Pre-quantized gamma in Q11 format (clamp(gamma, -16, 16) * 2047)
        Q8_1Block *output,
        std::size_t blocks_per_row,
        int32_t epsilon_scaled) // Epsilon pre-scaled appropriately
    {
        if (!input || !gamma_q12 || !output || blocks_per_row == 0)
            return;

        const size_t cols = blocks_per_row * 32;

        // ====================================================================
        // Phase 1 & 2: Compute Sum(d² * sum_qs²) using double precision
        // This avoids overflow/underflow issues with fixed-point accumulation
        // ====================================================================
        double total_sum_sq = 0.0;

        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            float d = fp16_to_fp32_q8(input[b].d);
            int32_t s = 0;

#if defined(__AVX512F__)
            __m256i qs_8 = _mm256_loadu_si256((const __m256i *)input[b].qs);
            __m512i qs_16 = _mm512_cvtepi8_epi16(qs_8);
            __m512i qs_sq = _mm512_madd_epi16(qs_16, qs_16);
            s = _mm512_reduce_add_epi32(qs_sq);
#else
            for (int i = 0; i < 32; ++i)
                s += input[b].qs[i] * input[b].qs[i];
#endif
            total_sum_sq += static_cast<double>(d) * d * s;
        }

        // ====================================================================
        // Phase 3: Compute RMS and inverse RMS
        // ====================================================================

        // Mean = Total / Cols
        double mean_sq = total_sum_sq / cols;
        mean_sq += 1e-5; // Epsilon

        // Compute 1/sqrt(mean_qs_sq) in Q24 format
        // inv_rms = 2^24 / sqrt(mean_sq)
        double rms = std::sqrt(mean_sq);
        double inv_rms_f = 16777216.0 / rms;

        // Clamp to uint32 range to avoid overflow in integer loop
        if (inv_rms_f > 2147483647.0)
            inv_rms_f = 2147483647.0;

        uint32_t inv_rms_q24 = static_cast<uint32_t>(inv_rms_f);
        if (inv_rms_q24 == 0)
            inv_rms_q24 = 1;

        // ====================================================================
        // Phase 4: Apply normalization (vectorized)
        // For each block:
        //   1) Compute y_high = qs * gamma * inv_rms (64-bit precision)
        //      y_high scale: Q0 * Q12 * (2^14/RMS) = Q26 / RMS
        //   2) y = y_high >> 20 (Q6 / RMS)
        //   3) Find max_abs = max(|y|) across the block
        //   4) d_out = d_in * max_abs / (127 * 64)
        //   5) qs_out = clamp(y * 127 / max_abs, -127, 127)
        // ====================================================================

#if defined(__AVX512F__)
        __m512i v_inv_rms = _mm512_set1_epi32(inv_rms_q24);
        // Permutation indices for interleaving even/odd results
        __m512i v_perm_idx = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);

        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            const Q8_1Block &in_blk = input[b];
            Q8_1Block &out_blk = output[b];
            float d_in = fp16_to_fp32_q8(in_blk.d);

            // Load and expand qs to int32
            __m128i qs_lo = _mm_loadu_si128((const __m128i *)in_blk.qs);
            __m128i qs_hi = _mm_loadu_si128((const __m128i *)(in_blk.qs + 16));
            __m512i qs32_lo = _mm512_cvtepi8_epi32(qs_lo);
            __m512i qs32_hi = _mm512_cvtepi8_epi32(qs_hi);

            // Load gamma (int16) -> int32
            __m256i gamma16_lo = _mm256_loadu_si256((const __m256i *)(gamma_q12 + b * 32));
            __m256i gamma16_hi = _mm256_loadu_si256((const __m256i *)(gamma_q12 + b * 32 + 16));
            __m512i gamma32_lo = _mm512_cvtepi16_epi32(gamma16_lo);
            __m512i gamma32_hi = _mm512_cvtepi16_epi32(gamma16_hi);

            // Compute term = qs * gamma (int32)
            __m512i term_lo = _mm512_mullo_epi32(qs32_lo, gamma32_lo);
            __m512i term_hi = _mm512_mullo_epi32(qs32_hi, gamma32_hi);

            // Compute res = term * inv_rms (64-bit)
            // Even lanes
            __m512i res_lo_even = _mm512_mul_epi32(term_lo, v_inv_rms);
            __m512i res_hi_even = _mm512_mul_epi32(term_hi, v_inv_rms);
            // Odd lanes (shift term right by 32)
            __m512i res_lo_odd = _mm512_mul_epi32(_mm512_srli_epi64(term_lo, 32), v_inv_rms);
            __m512i res_hi_odd = _mm512_mul_epi32(_mm512_srli_epi64(term_hi, 32), v_inv_rms);

            // Shift right by 20 to get Q6 result (fits in int32)
            __m512i y_lo_even = _mm512_srai_epi64(res_lo_even, 20);
            __m512i y_hi_even = _mm512_srai_epi64(res_hi_even, 20);
            __m512i y_lo_odd = _mm512_srai_epi64(res_lo_odd, 20);
            __m512i y_hi_odd = _mm512_srai_epi64(res_hi_odd, 20);

            // Pack back to int32
            __m256i y256_lo_even = _mm512_cvtepi64_epi32(y_lo_even);
            __m256i y256_lo_odd = _mm512_cvtepi64_epi32(y_lo_odd);
            __m256i y256_hi_even = _mm512_cvtepi64_epi32(y_hi_even);
            __m256i y256_hi_odd = _mm512_cvtepi64_epi32(y_hi_odd);

            // Interleave even/odd results
            __m512i y_lo = _mm512_castsi256_si512(y256_lo_even);
            y_lo = _mm512_inserti64x4(y_lo, y256_lo_odd, 1);
            y_lo = _mm512_permutexvar_epi32(v_perm_idx, y_lo);

            __m512i y_hi = _mm512_castsi256_si512(y256_hi_even);
            y_hi = _mm512_inserti64x4(y_hi, y256_hi_odd, 1);
            y_hi = _mm512_permutexvar_epi32(v_perm_idx, y_hi);

            // Find max absolute value
            __m512i abs_lo = _mm512_abs_epi32(y_lo);
            __m512i abs_hi = _mm512_abs_epi32(y_hi);
            __m512i abs_max_vec = _mm512_max_epi32(abs_lo, abs_hi);
            int32_t max_abs = _mm512_reduce_max_epi32(abs_max_vec);

            // Compute output scale: d_out = d_in * max_abs / (127 * 65536)
            // 65536 comes from the Q16 nature of y (result of >> 20 from Q36)
            float d_out;
            if (max_abs <= 127)
            {
                if (max_abs == 0)
                {
                    d_out = 0.0f;
                }
                else
                {
                    d_out = d_in * static_cast<float>(max_abs) / (127.0f * 32768.0f);
                }
            }
            else
            {
                d_out = d_in * static_cast<float>(max_abs) / (127.0f * 32768.0f);
            }

            // Rescale y values: y_rescaled = y * 127 / max_abs
            if (max_abs > 0)
            {
                __m512 scale_factor = _mm512_set1_ps(127.0f / static_cast<float>(max_abs));
                __m512 y_f32_lo = _mm512_cvtepi32_ps(y_lo);
                __m512 y_f32_hi = _mm512_cvtepi32_ps(y_hi);
                y_f32_lo = _mm512_mul_ps(y_f32_lo, scale_factor);
                y_f32_hi = _mm512_mul_ps(y_f32_hi, scale_factor);
                y_lo = _mm512_cvtps_epi32(y_f32_lo);
                y_hi = _mm512_cvtps_epi32(y_f32_hi);
            }

            // Clamp to [-127, 127]
            __m512i v_127 = _mm512_set1_epi32(127);
            __m512i v_neg127 = _mm512_set1_epi32(-127);
            y_lo = _mm512_max_epi32(y_lo, v_neg127);
            y_lo = _mm512_min_epi32(y_lo, v_127);
            y_hi = _mm512_max_epi32(y_hi, v_neg127);
            y_hi = _mm512_min_epi32(y_hi, v_127);

            // Pack back to int8
            __m128i out8_lo = _mm512_cvtepi32_epi8(y_lo);
            __m128i out8_hi = _mm512_cvtepi32_epi8(y_hi);

            _mm_storeu_si128((__m128i *)out_blk.qs, out8_lo);
            _mm_storeu_si128((__m128i *)(out_blk.qs + 16), out8_hi);

            // Compute sum_qs
            int32_t sum_lo = _mm512_reduce_add_epi32(y_lo);
            int32_t sum_hi = _mm512_reduce_add_epi32(y_hi);
            out_blk.sum_qs = static_cast<int16_t>(sum_lo + sum_hi);

            // Store output scale
            out_blk.d = fp32_to_fp16_q8(d_out);
        }
#else
        // Scalar fallback
        for (size_t b = 0; b < blocks_per_row; ++b)
        {
            const Q8_1Block &in_blk = input[b];
            Q8_1Block &out_blk = output[b];
            float d_in = fp16_to_fp32_q8(in_blk.d);

            // First pass: compute raw y values and find max_abs
            int32_t y_raw[32];
            int32_t max_abs = 0;
            for (int i = 0; i < 32; ++i)
            {
                size_t ch = b * 32 + i;
                // High precision computation: qs * gamma * inv_rms
                int64_t term = static_cast<int64_t>(in_blk.qs[i]) * gamma_q12[ch];
                int64_t res = term * inv_rms_q24;
                y_raw[i] = static_cast<int32_t>(res >> 20);
                max_abs = std::max(max_abs, std::abs(y_raw[i]));
            }

            // Compute output scale
            float d_out;
            float scale_factor = 1.0f;
            if (max_abs == 0)
            {
                d_out = 0.0f;
                scale_factor = 0.0f;
            }
            else
            {
                d_out = d_in * static_cast<float>(max_abs) / (127.0f * 32768.0f);
                scale_factor = 127.0f / static_cast<float>(max_abs);
            }

            // Second pass: rescale and clamp
            int32_t sum_qs = 0;
            for (int i = 0; i < 32; ++i)
            {
                int32_t y = static_cast<int32_t>(std::round(y_raw[i] * scale_factor));
                y = (y > 127) ? 127 : (y < -127) ? -127
                                                 : y;
                out_blk.qs[i] = static_cast<int8_t>(y);
                sum_qs += y;
            }
            out_blk.sum_qs = static_cast<int16_t>(sum_qs);
            out_blk.d = fp32_to_fp16_q8(d_out);
        }
#endif
    }

    void rmsnorm_q8_1_pure_integer(
        const Q8_1Block *input,
        const float *gamma,
        Q8_1Block *output,
        std::size_t rows,
        std::size_t blocks_per_row,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        if (!input || !gamma || !output || rows == 0 || blocks_per_row == 0)
            return;

        const size_t cols = blocks_per_row * 32;

        // Scale epsilon to integer domain
        // Typical epsilon is 1e-5 to 1e-6. We scale to ~1-100 range.
        int32_t epsilon_scaled = static_cast<int32_t>(epsilon * 1e10f);
        if (epsilon_scaled < 1)
            epsilon_scaled = 1;

        // Get parallelization thresholds from DebugEnv
        const auto &cfg = debugEnv().rmsnorm;

        // ================================================================
        // UNIFIED ROW-PARALLEL APPROACH
        // 1. Quantize gamma once (sequential, ~1% of work)
        // 2. Process rows in parallel (each row: sumsq -> inv_rms -> apply)
        //
        // This is simpler than 3-phase and avoids intermediate allocations
        // for row_sumsq/row_inv_rms vectors. Each thread does complete
        // per-row processing, which has better cache locality.
        // ================================================================

        // Phase 0: Quantize gamma ONCE (sequential, shared across all rows)
        // Use thread-local cache to avoid repeated allocations across calls
        // Cache key: gamma pointer + cols (assumes same gamma pointer = same weights)
        thread_local std::vector<int16_t> gamma_q12_cache;
        thread_local const float *cached_gamma_ptr = nullptr;
        thread_local size_t cached_cols = 0;

        int16_t *gamma_q12 = nullptr;

        if (cached_gamma_ptr == gamma && cached_cols == cols)
        {
            // Cache hit - reuse previous quantization
            gamma_q12 = gamma_q12_cache.data();
        }
        else
        {
            // Cache miss - quantize gamma and update cache
            if (gamma_q12_cache.size() < cols)
            {
                gamma_q12_cache.resize(cols);
            }
            gamma_q12 = gamma_q12_cache.data();

            // NOTE: Using Q11 format with [-16, 16] range to support larger gamma values
            // found in Qwen2.5 models (some gamma weights exceed 13.0).
            // Previous Q12 format with [-8, 8] range caused ~38% error on these elements.
            // Scale factor: 2047 (11 fractional bits) instead of 4095 (12 fractional bits)
#ifdef __AVX512F__
            {
                size_t i = 0;
                const __m512 scale = _mm512_set1_ps(2047.0f); // Q11 scale for [-16, 16] range
                const __m512 clamp_lo = _mm512_set1_ps(-16.0f);
                const __m512 clamp_hi = _mm512_set1_ps(16.0f);
                const __m512 int16_lo = _mm512_set1_ps(-32767.0f);
                const __m512 int16_hi = _mm512_set1_ps(32767.0f);

                for (; i + 32 <= cols; i += 32)
                {
                    __m512 v0 = _mm512_loadu_ps(gamma + i);
                    __m512 v1 = _mm512_loadu_ps(gamma + i + 16);
                    v0 = _mm512_max_ps(v0, clamp_lo);
                    v0 = _mm512_min_ps(v0, clamp_hi);
                    v1 = _mm512_max_ps(v1, clamp_lo);
                    v1 = _mm512_min_ps(v1, clamp_hi);
                    v0 = _mm512_mul_ps(v0, scale);
                    v1 = _mm512_mul_ps(v1, scale);
                    v0 = _mm512_max_ps(v0, int16_lo);
                    v0 = _mm512_min_ps(v0, int16_hi);
                    v1 = _mm512_max_ps(v1, int16_lo);
                    v1 = _mm512_min_ps(v1, int16_hi);
                    __m512i i0 = _mm512_cvtps_epi32(v0);
                    __m512i i1 = _mm512_cvtps_epi32(v1);
                    __m512i packed = _mm512_packs_epi32(i0, i1);
                    packed = _mm512_permutexvar_epi64(
                        _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), packed);
                    _mm512_storeu_si512(reinterpret_cast<__m512i *>(gamma_q12 + i), packed);
                }
                for (; i < cols; ++i)
                {
                    float clamped = std::max(-16.0f, std::min(16.0f, gamma[i]));
                    float scaled = clamped * 2047.0f;
                    scaled = std::max(-32767.0f, std::min(32767.0f, scaled));
                    gamma_q12[i] = static_cast<int16_t>(scaled);
                }
            }
#else
            for (size_t i = 0; i < cols; ++i)
            {
                float clamped = std::max(-16.0f, std::min(16.0f, gamma[i]));
                float scaled = clamped * 2047.0f;
                scaled = std::max(-32767.0f, std::min(32767.0f, scaled));
                gamma_q12[i] = static_cast<int16_t>(scaled);
            }
#endif
            // Update cache metadata
            cached_gamma_ptr = gamma;
            cached_cols = cols;
        }

        // Process rows - parallelize if we have enough work
        // Each row does: sumsq -> inv_rms -> apply (all in rmsnorm_q8_1_pure_integer_row)
        // ================================================================
        // SMOOTH DYNAMIC THREADING (Dec 2025)
        //
        // Single heuristic: threads = total_bytes / bytes_per_thread
        // This naturally scales from 1 thread (small) to many (large).
        // No hard cutoffs - the math handles it smoothly.
        // ================================================================
#ifdef _OPENMP
        bool should_parallelize = opts.allow_parallel;

        if (should_parallelize)
        {
            // Use OMP_WORKSHARE_REGION for nested-safe execution
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (long long r = 0; r < static_cast<long long>(rows); ++r)
                {
                    rmsnorm_q8_1_pure_integer_row_unrolled(
                        input + r * blocks_per_row,
                        gamma_q12,
                        output + r * blocks_per_row,
                        blocks_per_row,
                        epsilon_scaled);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
            return;
        }
#endif
        // Sequential path
        for (size_t r = 0; r < rows; ++r)
        {
            rmsnorm_q8_1_pure_integer_row_unrolled(
                input + r * blocks_per_row,
                gamma_q12,
                output + r * blocks_per_row,
                blocks_per_row,
                epsilon_scaled);
        }
    }

    // ========================================================================
    // Q16_1 RMSNorm (Q16_1 input → FP32 output)
    // ========================================================================

    void rmsnorm_q16_1_fp32_row_scalar(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon)
    {
        if (!input || !gamma || !output || cols == 0)
            return;

        const size_t n_blocks = cols / 32;

        // Pass 1: Compute sum of squares with double precision accumulation
        // Dequantize on-the-fly: val = scale * qs
        double sum_sq = 0.0;
        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            float scale = block.d;

            for (int i = 0; i < 32; ++i)
            {
                float val = scale * static_cast<float>(block.qs[i]);
                sum_sq += static_cast<double>(val * val);
            }
        }

        // Compute RMS and inverse
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        float inv_rms = 1.0f / rms;

        // Pass 2: Dequantize + normalize + gamma + write
        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            float scale = block.d;

            for (int i = 0; i < 32; ++i)
            {
                size_t idx = blk * 32 + i;
                float val = scale * static_cast<float>(block.qs[i]);
                output[idx] = val * inv_rms * gamma[idx];
            }
        }
    }

#if defined(__AVX512F__)
    void rmsnorm_q16_1_fp32_row_avx512(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon)
    {
        if (!input || !gamma || !output || cols == 0)
            return;

        const size_t n_blocks = cols / 32;

        // Pass 1: Vectorized sum of squares
        // Use double precision accumulation for numerical stability
        __m512d sum_sq_d_lo = _mm512_setzero_pd();
        __m512d sum_sq_d_hi = _mm512_setzero_pd();

        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            __m512 scale_vec = _mm512_set1_ps(block.d);

            // Load 16 int16 elements at a time → expand to 16 int32 → convert to 16 FP32
            // First half: qs[0..15]
            __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));
            __m512i q32_lo = _mm512_cvtepi16_epi32(q_lo);
            __m512 f_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(q32_lo), scale_vec);

            // Second half: qs[16..31]
            __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs + 16));
            __m512i q32_hi = _mm512_cvtepi16_epi32(q_hi);
            __m512 f_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(q32_hi), scale_vec);

            // Compute squares (FP32)
            __m512 sq_lo = _mm512_mul_ps(f_lo, f_lo);
            __m512 sq_hi = _mm512_mul_ps(f_hi, f_hi);

            // Convert to double and accumulate (lower 8 floats → 8 doubles)
            __m256 sq_lo_lo = _mm512_castps512_ps256(sq_lo);
            __m256 sq_lo_hi = _mm512_extractf32x8_ps(sq_lo, 1);
            __m256 sq_hi_lo = _mm512_castps512_ps256(sq_hi);
            __m256 sq_hi_hi = _mm512_extractf32x8_ps(sq_hi, 1);

            sum_sq_d_lo = _mm512_add_pd(sum_sq_d_lo, _mm512_cvtps_pd(sq_lo_lo));
            sum_sq_d_hi = _mm512_add_pd(sum_sq_d_hi, _mm512_cvtps_pd(sq_lo_hi));
            sum_sq_d_lo = _mm512_add_pd(sum_sq_d_lo, _mm512_cvtps_pd(sq_hi_lo));
            sum_sq_d_hi = _mm512_add_pd(sum_sq_d_hi, _mm512_cvtps_pd(sq_hi_hi));
        }

        // Horizontal sum of doubles
        __m512d sum_total_d = _mm512_add_pd(sum_sq_d_lo, sum_sq_d_hi);
        double sum_sq = _mm512_reduce_add_pd(sum_total_d);

        // Compute RMS and inverse
        float rms = std::sqrt(static_cast<float>(sum_sq / cols) + epsilon);
        __m512 inv_rms_vec = _mm512_set1_ps(1.0f / rms);

        // Pass 2: Vectorized dequant + normalize + gamma + write
        for (size_t blk = 0; blk < n_blocks; ++blk)
        {
            const Q16_1Block &block = input[blk];
            __m512 scale_vec = _mm512_set1_ps(block.d);

            // Dequantize: qs → FP32
            __m256i q_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));
            __m256i q_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs + 16));

            __m512 f_lo = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_lo)), scale_vec);
            __m512 f_hi = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(q_hi)), scale_vec);

            // Load gamma
            __m512 g_lo = _mm512_loadu_ps(gamma + blk * 32);
            __m512 g_hi = _mm512_loadu_ps(gamma + blk * 32 + 16);

            // Normalize and apply gamma: output = (val * inv_rms) * gamma
            __m512 out_lo = _mm512_mul_ps(_mm512_mul_ps(f_lo, inv_rms_vec), g_lo);
            __m512 out_hi = _mm512_mul_ps(_mm512_mul_ps(f_hi, inv_rms_vec), g_hi);

            // Store result
            _mm512_storeu_ps(output + blk * 32, out_lo);
            _mm512_storeu_ps(output + blk * 32 + 16, out_hi);
        }
    }
#else
    void rmsnorm_q16_1_fp32_row_avx512(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t cols,
        float epsilon)
    {
        // Fallback to scalar on non-AVX512 systems
        rmsnorm_q16_1_fp32_row_scalar(input, gamma, output, cols, epsilon);
    }
#endif

    void rmsnorm_q16_1_fp32_fused(
        const Q16_1Block *input,
        const float *gamma,
        float *output,
        std::size_t rows,
        std::size_t cols,
        float epsilon,
        const RMSNormExecOptions &opts)
    {
        if (!input || !gamma || !output || rows == 0 || cols == 0)
            return;

        const size_t blocks_per_row = cols / 32;

        // Select implementation based on CPU features
#if defined(__AVX512F__)
        auto row_fn = rmsnorm_q16_1_fp32_row_avx512;
#else
        auto row_fn = rmsnorm_q16_1_fp32_row_scalar;
#endif

        if (opts.force_scalar)
        {
            row_fn = rmsnorm_q16_1_fp32_row_scalar;
        }

#ifdef _OPENMP
        bool should_parallelize = opts.allow_parallel && (rows * cols >= opts.parallel_threshold_elems);

        if (should_parallelize)
        {
            auto do_work = [&]()
            {
#pragma omp for schedule(static)
                for (long long r = 0; r < static_cast<long long>(rows); ++r)
                {
                    row_fn(
                        input + r * blocks_per_row,
                        gamma,
                        output + r * cols,
                        cols,
                        epsilon);
                }
            };
            OMP_WORKSHARE_REGION(do_work);
            return;
        }
#endif

        // Sequential path
        for (size_t r = 0; r < rows; ++r)
        {
            row_fn(
                input + r * blocks_per_row,
                gamma,
                output + r * cols,
                cols,
                epsilon);
        }
    }

} // namespace llaminar2::primitives