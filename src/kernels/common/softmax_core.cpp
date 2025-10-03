/**
 * @file softmax_core.cpp
 * @brief Implementation of unified softmax primitives
 * @author David Sanftenberg
 */
#include "softmax_core.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include "../../utils/debug_env.h"
#if defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace llaminar::kernels
{

    namespace
    {
        inline float mask_val(bool masked, float v)
        {
            return masked ? -std::numeric_limits<float>::infinity() : v;
        }
#if defined(__AVX2__) && !defined(__AVX512F__)
        // Horizontal max reduction for AVX2 register without spilling to memory.
        static inline float hmax256(__m256 v)
        {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 m = _mm_max_ps(lo, hi);
            // Fold 4 -> 2
            __m128 shuf = _mm_movehdup_ps(m); // (b,d) duplicates high lanes
            m = _mm_max_ps(m, shuf);
            shuf = _mm_movehl_ps(shuf, m); // high two lanes
            m = _mm_max_ps(m, shuf);
            // Now lowest lane has max
            return _mm_cvtss_f32(m);
        }
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
        // Approximate exp for softmax acceleration (optional, enabled via LLAMINAR_SOFTMAX_FAST_EXP)
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
    }

    void softmax_row_major(const SoftmaxRowArgs &args)
    {
        if (!args.scores || args.rows <= 0 || args.cols <= 0)
            return;

        float *scores = args.scores;
        const int rows = args.rows;
        const int cols = args.cols;
        const bool causal = args.causal;
        const float scale = args.scale;

        const auto &env = llaminar::debugEnv().softmax;
        bool force_scalar = env.force_scalar;
        const bool use_fast_exp = env.fast_exp;
        const bool validate_fast = env.validate && use_fast_exp; // placeholder for future deeper validation
        // Heuristic: parallelize over rows when we have enough rows or total elements.
        std::size_t total_elems = (std::size_t)rows * (std::size_t)cols;
        bool parallel = !force_scalar && rows > 1 && total_elems >= (std::size_t)env.parallel_elems_threshold;
        if (env.parallel_row_threshold > 0 && rows < env.parallel_row_threshold)
            parallel = false;
#ifdef _OPENMP
        if (omp_in_parallel())
            parallel = false; // don't nest
#endif

#pragma omp parallel for if (parallel)
        for (long long r = 0; r < (long long)rows; ++r)
        {
            float *row = scores + (std::size_t)r * cols;
            float row_max = -std::numeric_limits<float>::infinity();
            // Pass 1: find max (with causal masking inline)
            // Vectorized max pass
#if defined(__AVX512F__)
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
                    // Number of valid (unmasked) lanes = clamp(r - c + 1, 0, 16)
                    int n_valid = (int)r - c + 1;
                    if (n_valid <= 0)
                    {
                        // All masked
                        vmax = _mm512_max_ps(vmax, neg_inf);
                        continue;
                    }
                    if (n_valid < 16)
                    {
                        __mmask16 kmask = (__mmask16)((1u << n_valid) - 1u);
                        // Move only valid lanes; others become -inf
                        __m512 vmasked = _mm512_mask_mov_ps(neg_inf, kmask, v);
                        vmax = _mm512_max_ps(vmax, vmasked);
                        continue;
                    }
                    // n_valid == 16; full vector valid
                }
                vmax = _mm512_max_ps(vmax, v);
            }
            // Horizontal reduction using AVX-512 intrinsic (avoid spill to stack)
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
#elif defined(__AVX2__)
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
                        int gc = c + lane;
                        if (gc > r)
                            tmp[lane] = -std::numeric_limits<float>::infinity();
                    }
                    v = _mm256_loadu_ps(tmp);
                }
                vmax = _mm256_max_ps(vmax, v);
            }
            // Horizontal reduction (intrinsic sequence) instead of memory spill
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
#else
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
#endif
            if (!std::isfinite(row_max))
                row_max = 0.0f;
            // New fused approach: second pass only accumulates sum of exp((x*scale - max)), does not write.
            // Third pass computes normalized values directly (one write total instead of write+scale rewrite).
            double sum = 0.0;
#if defined(__AVX512F__)
            int c2 = 0;
            const __m512 row_max_ps = _mm512_set1_ps(row_max);
            const __m512 scale_ps = _mm512_set1_ps(scale);
            for (; c2 + 16 <= cols; c2 += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c2);
                if (scale != 1.0f)
                    v = _mm512_mul_ps(v, scale_ps);
                int n_valid = causal ? ((int)r - c2 + 1) : 16;
                if (n_valid <= 0)
                    continue; // all masked contributes nothing
                __mmask16 kmask = (n_valid >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << n_valid) - 1u);
                alignas(64) float tmp[16];
                _mm512_store_ps(tmp, v);
                for (int lane = 0; lane < 16; ++lane)
                {
                    if (!(kmask & (1u << lane)))
                        continue;
                    float dv = tmp[lane] - row_max;
                    sum += std::exp(dv);
                }
            }
            for (; c2 < cols; ++c2)
            {
                bool masked = causal && c2 > r;
                if (masked)
                    continue;
                float v = row[c2];
                if (scale != 1.0f)
                    v *= scale;
                sum += std::exp(v - row_max);
            }
#elif defined(__AVX2__)
            int c2 = 0;
            __m256 scale_ps = _mm256_set1_ps(scale);
            for (; c2 + 8 <= cols; c2 += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c2);
                if (scale != 1.0f)
                    v = _mm256_mul_ps(v, scale_ps);
                float tmp[8];
                _mm256_storeu_ps(tmp, v);
                for (int lane = 0; lane < 8; ++lane)
                {
                    int gc = c2 + lane;
                    if (causal && gc > r)
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
#else
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
#endif
            if (sum <= 0.0)
                sum = 1.0;
            float inv = (float)(1.0 / sum);
            // Third pass: write normalized values directly.
#if defined(__AVX512F__)
            int c3 = 0;
            const __m512 inv_ps = _mm512_set1_ps(inv);
            for (; c3 + 16 <= cols; c3 += 16)
            {
                __m512 v = _mm512_loadu_ps(row + c3);
                if (scale != 1.0f)
                    v = _mm512_mul_ps(v, scale_ps); // reuse scale_ps
                int n_valid = causal ? ((int)r - c3 + 1) : 16;
                if (n_valid <= 0)
                {
                    _mm512_storeu_ps(row + c3, _mm512_setzero_ps());
                    continue;
                }
                __mmask16 kmask = (n_valid >= 16) ? (__mmask16)0xFFFF : (__mmask16)((1u << n_valid) - 1u);
                alignas(64) float tmp[16];
                _mm512_store_ps(tmp, v);
                float out[16];
                if (use_fast_exp)
                {
                    __m512 raw = _mm512_loadu_ps(tmp);
                    __m512 shifted = _mm512_sub_ps(raw, _mm512_set1_ps(row_max));
                    __m512 ex = fast_exp_approx512(shifted);
                    __m512 scaled = _mm512_mul_ps(ex, _mm512_set1_ps(inv));
                    alignas(64) float tmp_out[16];
                    _mm512_store_ps(tmp_out, scaled);
                    for (int lane = 0; lane < 16; ++lane)
                    {
                        if (!(kmask & (1u << lane)))
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        out[lane] = tmp_out[lane];
                    }
                }
                else
                {
                    for (int lane = 0; lane < 16; ++lane)
                    {
                        if (!(kmask & (1u << lane)))
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        out[lane] = (float)(std::exp(tmp[lane] - row_max) * inv);
                    }
                }
                _mm512_storeu_ps(row + c3, _mm512_loadu_ps(out));
            }
            for (; c3 < cols; ++c3)
            {
                bool masked = causal && c3 > r;
                if (masked)
                {
                    row[c3] = 0.f;
                    continue;
                }
                float v = row[c3];
                if (scale != 1.0f)
                    v *= scale;
                float dv = v - row_max;
                if (use_fast_exp)
                {
#if defined(__AVX512F__)
                    __m512 x = _mm512_set1_ps(dv);
                    __m512 ex = fast_exp_approx512(x);
                    alignas(64) float t[16];
                    _mm512_store_ps(t, ex);
                    row[c3] = t[0] * inv;
#else
                    row[c3] = std::exp(dv) * inv;
#endif
                }
                else
                {
                    row[c3] = std::exp(dv) * inv;
                }
            }
#elif defined(__AVX2__)
            int c3 = 0;
            for (; c3 + 8 <= cols; c3 += 8)
            {
                __m256 v = _mm256_loadu_ps(row + c3);
                if (scale != 1.0f)
                    v = _mm256_mul_ps(v, scale_ps);
                float tmp[8];
                _mm256_storeu_ps(tmp, v);
                float out[8];
                if (use_fast_exp)
                {
                    __m256 raw = _mm256_loadu_ps(tmp);
                    __m256 shifted = _mm256_sub_ps(raw, _mm256_set1_ps(row_max));
                    __m256 ex = fast_exp_approx256(shifted);
                    __m256 scaled = _mm256_mul_ps(ex, _mm256_set1_ps(inv));
                    float tmp_out[8];
                    _mm256_storeu_ps(tmp_out, scaled);
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        int gc = c3 + lane;
                        if (causal && gc > r)
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        out[lane] = tmp_out[lane];
                    }
                }
                else
                {
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        int gc = c3 + lane;
                        if (causal && gc > r)
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        float dv = tmp[lane] - row_max;
                        float e = std::exp(dv);
                        out[lane] = e * inv;
                    }
                }
                _mm256_storeu_ps(row + c3, _mm256_loadu_ps(out));
            }
            for (; c3 < cols; ++c3)
            {
                if (causal && c3 > r)
                {
                    row[c3] = 0.f;
                    continue;
                }
                float v = row[c3];
                if (scale != 1.0f)
                    v *= scale;
                float dv2 = v - row_max;
                if (use_fast_exp)
                {
#if defined(__AVX2__)
                    __m256 x = _mm256_set1_ps(dv2);
                    __m256 ex = fast_exp_approx256(x);
                    float t[8];
                    _mm256_storeu_ps(t, ex);
                    row[c3] = t[0] * inv;
#else
                    row[c3] = std::exp(dv2) * inv;
#endif
                }
                else
                {
                    row[c3] = std::exp(dv2) * inv;
                }
            }
#else
#pragma omp simd
            for (long long c = 0; c < (long long)cols; ++c)
            {
                bool masked = causal && c > r;
                float v = row[c];
                if (scale != 1.0f)
                    v *= scale;
                if (masked)
                {
                    row[c] = 0.f;
                    continue;
                }
                row[c] = (float)(std::exp(v - row_max) * inv);
            }
#endif
        }
    }

    void softmax_distributed(const SoftmaxRowArgs &local_args,
                             int global_rows,
                             int row_offset,
                             int global_cols,
                             int local_col_offset,
                             const DistributedSoftmaxCtx &ctx)
    {
        if (!local_args.scores || local_args.rows < 0 || local_args.cols <= 0 || global_rows <= 0 || global_cols <= 0)
            return;
        if (ctx.comm == MPI_COMM_NULL)
        {
            // Fallback to local only if communicator invalid
            softmax_row_major(local_args);
            return;
        }

        const int local_rows = local_args.rows;
        const int local_cols = local_args.cols;
        float *scores = local_args.scores;
        const bool causal = local_args.causal;
        const float scale = local_args.scale;

        // Allocate per-row buffers (stack vectors) for row maxima & denominators local/global.
        std::vector<float> local_max(local_rows, -std::numeric_limits<float>::infinity());
        std::vector<float> global_max(local_rows, 0.f);
        std::vector<double> local_sum(local_rows, 0.0);
        std::vector<double> global_sum(local_rows, 0.0);

        // Heuristic: row-parallel inside each rank (rank-level parallelism already present across MPI ranks).
        const auto &env = llaminar::debugEnv().softmax;
        bool force_scalar = env.force_scalar;
        std::size_t total_elems = (std::size_t)local_rows * (std::size_t)local_cols;
        bool parallel = !force_scalar && local_rows > 1 && total_elems >= (std::size_t)env.parallel_elems_threshold;
#ifdef _OPENMP
        if (omp_in_parallel())
            parallel = false;
#endif

        // 1. Local max pass with causal masking (row-parallel) .
#pragma omp parallel for if (parallel)
        for (int r = 0; r < local_rows; ++r)
        {
            int global_r = row_offset + r;
            float *row = scores + static_cast<std::size_t>(r) * local_cols;
            float m = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < local_cols; ++c)
            {
                int global_c = local_col_offset + c;
                bool masked = causal && global_c > global_r;
                if (masked)
                    continue; // contributes nothing
                float v = row[c];
                if (scale != 1.f)
                    v *= scale;
                m = std::max(m, v);
            }
            local_max[r] = m;
        }

        // 2. Allreduce (MAX) across ranks for maxima.
        MPI_Allreduce(local_max.data(), global_max.data(), local_rows, MPI_FLOAT, MPI_MAX, ctx.comm);

        const bool fast_exp = env.fast_exp;
        bool recompute = env.dist_recompute; // explicit flag
        if (!env.dist_recompute && env.dist_recompute_threshold > 0)
        {
            std::size_t elems = (std::size_t)local_rows * (std::size_t)local_cols;
            if (elems >= (std::size_t)env.dist_recompute_threshold)
                recompute = true;
        }

        if (!recompute)
        {
#pragma omp parallel for if (parallel)
            for (int r = 0; r < local_rows; ++r)
            {
                float *row = scores + static_cast<std::size_t>(r) * local_cols;
                float gmax = global_max[r];
                if (!std::isfinite(gmax))
                    gmax = 0.f;
                double s = 0.0;
                int global_r = row_offset + r;
                int c = 0;
#if defined(__AVX512F__)
                for (; c + 16 <= local_cols; c += 16)
                {
                    __m512 v = _mm512_loadu_ps(row + c);
                    if (scale != 1.f)
                        v = _mm512_mul_ps(v, _mm512_set1_ps(scale));
                    // causal masking per lane
                    alignas(64) float tmp[16];
                    _mm512_store_ps(tmp, v);
                    float out[16];
                    for (int lane = 0; lane < 16; ++lane)
                    {
                        int gc = local_col_offset + c + lane;
                        bool masked = causal && gc > global_r;
                        if (masked)
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        float dv = tmp[lane] - gmax;
                        float e;
                        if (fast_exp)
                        {
                            __m512 x = _mm512_set1_ps(dv);
                            __m512 ex = fast_exp_approx512(x);
                            alignas(64) float t[16];
                            _mm512_store_ps(t, ex);
                            e = t[0];
                        }
                        else
                        {
                            e = std::exp(dv);
                        }
                        out[lane] = e;
                        s += e;
                    }
                    _mm512_storeu_ps(row + c, _mm512_loadu_ps(out));
                }
#elif defined(__AVX2__)
                for (; c + 8 <= local_cols; c += 8)
                {
                    __m256 v = _mm256_loadu_ps(row + c);
                    if (scale != 1.f)
                        v = _mm256_mul_ps(v, _mm256_set1_ps(scale));
                    float tmp[8];
                    _mm256_storeu_ps(tmp, v);
                    float out[8];
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        int gc = local_col_offset + c + lane;
                        bool masked = causal && gc > global_r;
                        if (masked)
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        float dv = tmp[lane] - gmax;
                        float e;
                        if (fast_exp)
                        {
                            __m256 x = _mm256_set1_ps(dv);
                            __m256 ex = fast_exp_approx256(x);
                            float t[8];
                            _mm256_storeu_ps(t, ex);
                            e = t[0];
                        }
                        else
                        {
                            e = std::exp(dv);
                        }
                        out[lane] = e;
                        s += e;
                    }
                    _mm256_storeu_ps(row + c, _mm256_loadu_ps(out));
                }
#endif
                for (; c < local_cols; ++c)
                {
                    int gc = local_col_offset + c;
                    bool masked = causal && gc > global_r;
                    if (masked)
                    {
                        row[c] = 0.f;
                        continue;
                    }
                    float v = row[c];
                    if (scale != 1.f)
                        v *= scale;
                    float dv = v - gmax;
                    float e = fast_exp ? std::exp(dv) : std::exp(dv); // scalar approx placeholder
                    row[c] = e;
                    s += e;
                }
                local_sum[r] = s;
            }
            MPI_Allreduce(local_sum.data(), global_sum.data(), local_rows, MPI_DOUBLE, MPI_SUM, ctx.comm);
#pragma omp parallel for if (parallel)
            for (int r = 0; r < local_rows; ++r)
            {
                double denom = global_sum[r];
                if (denom <= 0.0)
                    denom = 1.0;
                float inv = (float)(1.0 / denom);
                float *row = scores + static_cast<std::size_t>(r) * local_cols;
                for (int c = 0; c < local_cols; ++c)
                    row[c] *= inv;
            }
        }
        else
        {
            // Recompute path: sum exps first without storing them, then recompute normalized values.
#pragma omp parallel for if (parallel)
            for (int r = 0; r < local_rows; ++r)
            {
                float *row = scores + static_cast<std::size_t>(r) * local_cols;
                float gmax = global_max[r];
                if (!std::isfinite(gmax))
                    gmax = 0.f;
                double s = 0.0;
                int global_r = row_offset + r;
                int c = 0;
#if defined(__AVX512F__)
                for (; c + 16 <= local_cols; c += 16)
                {
                    __m512 v = _mm512_loadu_ps(row + c);
                    if (scale != 1.f)
                        v = _mm512_mul_ps(v, _mm512_set1_ps(scale));
                    alignas(64) float tmp[16];
                    _mm512_store_ps(tmp, v);
                    for (int lane = 0; lane < 16; ++lane)
                    {
                        int gc = local_col_offset + c + lane;
                        if (causal && gc > global_r)
                            continue;
                        float dv = tmp[lane] - gmax;
                        if (fast_exp)
                        {
                            __m512 x = _mm512_set1_ps(dv);
                            __m512 ex = fast_exp_approx512(x);
                            alignas(64) float t[16];
                            _mm512_store_ps(t, ex);
                            s += t[0];
                        }
                        else
                            s += std::exp(dv);
                    }
                }
#elif defined(__AVX2__)
                for (; c + 8 <= local_cols; c += 8)
                {
                    __m256 v = _mm256_loadu_ps(row + c);
                    if (scale != 1.f)
                        v = _mm256_mul_ps(v, _mm256_set1_ps(scale));
                    float tmp[8];
                    _mm256_storeu_ps(tmp, v);
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        int gc = local_col_offset + c + lane;
                        if (causal && gc > global_r)
                            continue;
                        float dv = tmp[lane] - gmax;
                        if (fast_exp)
                        {
                            __m256 x = _mm256_set1_ps(dv);
                            __m256 ex = fast_exp_approx256(x);
                            float t[8];
                            _mm256_storeu_ps(t, ex);
                            s += t[0];
                        }
                        else
                            s += std::exp(dv);
                    }
                }
#endif
                for (; c < local_cols; ++c)
                {
                    int gc = local_col_offset + c;
                    if (causal && gc > global_r)
                        continue;
                    float v = row[c];
                    if (scale != 1.f)
                        v *= scale;
                    float dv = v - gmax;
                    s += std::exp(dv); // keep precise for denominator in recompute path
                }
                local_sum[r] = s;
            }
            MPI_Allreduce(local_sum.data(), global_sum.data(), local_rows, MPI_DOUBLE, MPI_SUM, ctx.comm);
#pragma omp parallel for if (parallel)
            for (int r = 0; r < local_rows; ++r)
            {
                float *row = scores + static_cast<std::size_t>(r) * local_cols;
                float gmax = global_max[r];
                if (!std::isfinite(gmax))
                    gmax = 0.f;
                double denom = global_sum[r];
                if (denom <= 0.0)
                    denom = 1.0;
                float inv = (float)(1.0 / denom);
                int global_r = row_offset + r;
                int c = 0;
#if defined(__AVX512F__)
                for (; c + 16 <= local_cols; c += 16)
                {
                    __m512 raw = _mm512_loadu_ps(row + c);
                    if (scale != 1.f)
                        raw = _mm512_mul_ps(raw, _mm512_set1_ps(scale));
                    alignas(64) float tmp[16];
                    _mm512_store_ps(tmp, raw);
                    float out[16];
                    for (int lane = 0; lane < 16; ++lane)
                    {
                        int gc = local_col_offset + c + lane;
                        bool masked = causal && gc > global_r;
                        if (masked)
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        float dv = tmp[lane] - gmax;
                        if (fast_exp)
                        {
                            __m512 x = _mm512_set1_ps(dv);
                            __m512 ex = fast_exp_approx512(x);
                            alignas(64) float t[16];
                            _mm512_store_ps(t, ex);
                            out[lane] = t[0] * inv;
                        }
                        else
                            out[lane] = std::exp(dv) * inv;
                    }
                    _mm512_storeu_ps(row + c, _mm512_loadu_ps(out));
                }
#elif defined(__AVX2__)
                for (; c + 8 <= local_cols; c += 8)
                {
                    __m256 raw = _mm256_loadu_ps(row + c);
                    if (scale != 1.f)
                        raw = _mm256_mul_ps(raw, _mm256_set1_ps(scale));
                    float tmp[8];
                    _mm256_storeu_ps(tmp, raw);
                    float out[8];
                    for (int lane = 0; lane < 8; ++lane)
                    {
                        int gc = local_col_offset + c + lane;
                        bool masked = causal && gc > global_r;
                        if (masked)
                        {
                            out[lane] = 0.f;
                            continue;
                        }
                        float dv = tmp[lane] - gmax;
                        if (fast_exp)
                        {
                            __m256 x = _mm256_set1_ps(dv);
                            __m256 ex = fast_exp_approx256(x);
                            float t[8];
                            _mm256_storeu_ps(t, ex);
                            out[lane] = t[0] * inv;
                        }
                        else
                            out[lane] = std::exp(dv) * inv;
                    }
                    _mm256_storeu_ps(row + c, _mm256_loadu_ps(out));
                }
#endif
                for (; c < local_cols; ++c)
                {
                    int gc = local_col_offset + c;
                    bool masked = causal && gc > global_r;
                    if (masked)
                    {
                        row[c] = 0.f;
                        continue;
                    }
                    float v = row[c];
                    if (scale != 1.f)
                        v *= scale;
                    float dv = v - gmax;
                    row[c] = std::exp(dv) * inv; // precise scalar (approx tail optional)
                }
            }
        }
    }

} // namespace llaminar::kernels
